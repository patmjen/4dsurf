import sys
from itertools import product, permutations
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial import ConvexHull
import tetgen
import igl
import torch
from torch.utils.cpp_extension import load

# Load C extensions
if sys.platform == 'win32':
    _EXTRA_CFLAGS = ['/O2']
elif sys.platform == 'linux':
    _EXTRA_CFLAGS = ['-O3']
else:
    # Unknown system - do not specify any flags
    import warnings
    warnings.warn('Unknown system: no cflags specified for C extensions')
    _EXTRA_CFLAGS = []

c_tet_dihedral_angles = load(
    'tda',
    ['csrc/tet_dihedral_angles.cpp'],
    extra_cflags=_EXTRA_CFLAGS,
).tet_dihedral_angles

insert_tet_mesh_points = load(
    'itmp',
    [
        'csrc/insert_tet_mesh_points.cpp',
        'csrc/tetgen.cpp',
        'csrc/predicates.cpp'
    ],
    extra_cflags=_EXTRA_CFLAGS,
).insert_tet_mesh_points


#@torch.jit.script
def cross4(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Cross product of three 4D vectors u, v, and w.

    Args:
        u: (N, 4) tensor of coordinates for u vectors.
        v: (N, 4) tensor of coordinates for v vectors.
        w: (N, 4) tensor of coordinates for w vectors.

    Returns:
        normals: (N, 4) tensor of normal vectors.
    """
    # See: https://math.stackexchange.com/a/2371039
    u0, u1, u2, u3 = u.unbind(dim=1)
    v0, v1, v2, v3 = v.unbind(dim=1)
    w0, w1, w2, w3 = w.unbind(dim=1)

    # Precompute minors
    v2w1mv1w2 = v2 * w1 - v1 * w2
    v3w1mv1w3 = v3 * w1 - v1 * w3
    v3w2mv2w3 = v3 * w2 - v2 * w3
    v0w2mv2w0 = v0 * w2 - v2 * w0
    v0w3mv3w0 = v0 * w3 - v3 * w0
    v1w0mv0w1 = v1 * w0 - v0 * w1

    # Compute determinants
    a1 =  u3 * v2w1mv1w2 - u2 * v3w1mv1w3 + u1 * v3w2mv2w3
    a2 =  u3 * v0w2mv2w0 - u2 * v0w3mv3w0 - u0 * v3w2mv2w3
    a3 =  u3 * v1w0mv0w1 + u1 * v0w3mv3w0 + u0 * v3w1mv1w3
    a4 = -u2 * v1w0mv0w1 - u1 * v0w2mv2w0 - u0 * v2w1mv1w2

    return torch.stack([a1, a2, a3, a4], dim=1)


def sparse_diag(v: torch.Tensor) -> torch.Tensor:
    """
    Sparse diagonal matrix with v as diagonal.
    """
    nv = len(v)
    idx = torch.arange(nv)
    idx = torch.stack([idx, idx])
    return torch.sparse_coo_tensor(idx, v, (nv, nv))


def sparse_eye(n: int) -> torch.Tensor:
    """
    Sparse n x n identity matrix.
    """
    return sparse_diag(torch.ones(n))


#@torch.jit.script
def tet_normals(
    verts: torch.Tensor,
    tets: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Normal vectors for tetrahedra in 4D.

    Args:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.
        normalize: Whether to normalize vectors before returning.

    Returns:
        normals: (T, 4) tensor of normal vectors.
    """
    tet_verts = verts[tets]  # (T, 4, xyzt)
    u = tet_verts[:, 1] - tet_verts[:, 0]
    v = tet_verts[:, 2] - tet_verts[:, 0]
    w = tet_verts[:, 3] - tet_verts[:, 0]
    normals = cross4(u, v, w)
    if normalize:
        normals /= (torch.sum(normals**2, dim=1, keepdim=True).sqrt() + 1e-10)
    return normals


def tet_centers(
    verts: torch.Tensor,
    tets: torch.Tensor
) -> torch.Tensor:
    """
    Barycenters of tetrahedra.

    Args:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.

    Returns:
        centers: (T, 4) tensor of tet center coordinates.
    """
    return verts[tets].mean(dim=1)


def tet_volumes(
    verts: torch.Tensor,
    tets: torch.Tensor,
) -> torch.Tensor:
    """
    Volume of tetrahedra.

    Args:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.

    Returns:
        volumes: (T,) tensor of tet volumes.
    """
    v1, v2, v3, v4 = verts[tets].unbind(dim=1)  # Tet verts, each is (T, xyzt)

    # Compute edge vectors for tets
    u1 = v2 - v1
    u2 = v3 - v1
    u3 = v4 - v1

    # Compute Gram matrix entries
    g11 = torch.sum(u1 * u1, dim=1)
    g12 = torch.sum(u1 * u2, dim=1)
    g13 = torch.sum(u1 * u3, dim=1)
    g21 = torch.sum(u2 * u1, dim=1)
    g22 = torch.sum(u2 * u2, dim=1)
    g23 = torch.sum(u2 * u3, dim=1)
    g31 = torch.sum(u3 * u1, dim=1)
    g32 = torch.sum(u3 * u2, dim=1)
    g33 = torch.sum(u3 * u3, dim=1)

    # Compute Gram matrix determinants
    dets = g11 * (g22 * g33 - g23 * g32) \
         - g12 * (g21 * g33 - g23 * g31) \
         + g13 * (g21 * g32 - g22 * g31)

    dets = torch.clamp(dets, min=0)

    # Return tet volumes
    return dets.sqrt() / 6.0


def vert_normals(
    verts: torch.Tensor,
    tets: torch.Tensor,
    tet_norms: Optional[torch.Tensor] = None,
    normalize: bool = True,
    weight: str = 'average',
) -> torch.Tensor:
    """
    Args:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.
        tet_normals: Optional (T, 4) tensor of precomputed tet normal vectors.
        normalize: Whether to normalize vectors before returning.
        weight: Weights for tet normals. Must be 'average', 'angle', or 'area'.

    Returns:
        normals: (V, 4) tensor of normal vectors.
    """
    if weight not in ['average', 'angle', 'area']:
        raise ValueError(
            f"weight must be 'average', 'angle', or 'area' but was {weight}"
        )
    if tet_norms is None:
        tn = (weight != 'area')
        tet_norms = tet_normals(verts, tets, normalize=tn)  # (T, 4)
    tets_e = tets.view(-1, 4, 1).expand(-1, -1, 4)
    normals = torch.zeros_like(verts)
    if weight == 'average' or weight == 'area':
        for i in range(4):
            normals.scatter_add_(0, tets_e[:, i], tet_norms)
    elif weight == 'angle':
        angles = tet_solid_angles(verts, tets)  # (T, 4)
        for i in range(4):
            w = angles[:, i]
            normals.scatter_add_(0, tets_e[:, i], w[:, None] * tet_norms)

    if normalize:
        normals /= (torch.sum(normals**2, dim=1, keepdim=True).sqrt() + 1e-12)

    return normals


@torch.jit.script
def tet_edges(
    tets: torch.Tensor,
    return_tet_to_edge: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        tets: (T, 4) tensor of tet vertex indices.
        return_tet_to_edge: Return tet to edge index map

    Returns:
        edges: (E, 2) tensor of edges vertex indices.
        tet_to_edge: (T, 6) tensor of which element of edges correspond to each
            tet's 6 edges. Only returned if return_tet_to_edge is True.
    """
    # This code is heavily inspired by PyTorch3D
    # See: https://github.com/facebookresearch/pytorch3d/blob/49ed7b07b13be258a41904144854ed05aad5f60c/pytorch3d/structures/meshes.py#L1033
    max_idx = tets.max() + 1
    t0, t1, t2, t3 = tets.chunk(4, dim=1)

    e01 = torch.cat([t0, t1], dim=1)  # (T, 2)
    e02 = torch.cat([t0, t2], dim=1)  # (T, 2)
    e03 = torch.cat([t0, t3], dim=1)  # (T, 2)
    e12 = torch.cat([t1, t2], dim=1)  # (T, 2)
    e13 = torch.cat([t1, t3], dim=1)  # (T, 2)
    e23 = torch.cat([t2, t3], dim=1)  # (T, 2)
    edges = torch.cat([e01, e02, e03, e12, e13, e23])  # (6 * T, 2)
    edges, _ = edges.sort(dim=1)

    edges_hash = max_idx * edges[:, 0] + edges[:, 1]  # (6 * T, 1)
    if not return_tet_to_edge:
        u = torch.unique(edges_hash)
        return torch.stack([u // max_idx, u % max_idx], dim=1)
    else:
        u, inverse_indices = torch.unique(edges_hash, return_inverse=True)
        tet_to_edge = torch.stack(inverse_indices.chunk(6), dim=1)
        return torch.stack([u // max_idx, u % max_idx], dim=1), tet_to_edge


def tet_edge_lengths(verts: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    """
    Args:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.

    Returns:
        lengths: (E,) tensor of edge lengths.
    """
    edges = tet_edges(tets)
    v0, v1 = verts[edges].unbind(dim=1)
    return torch.sum((v1 - v0)**2, dim=-1).sqrt()


def tet_solid_angles(verts: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    """
    Compute solid angle for each tet vertex.

    Args:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.

    Returns:
        sangles: (T, 4) tensor of vertex solid angles.
    """
    da10, da20, da30, da21, da31, da32 = c_tet_dihedral_angles(verts, tets)

    # Compute solid angles and return
    return torch.stack([
        da10 + da20 + da30 - np.pi,
        da10 + da21 + da31 - np.pi,
        da20 + da21 + da32 - np.pi,
        da30 + da31 + da32 - np.pi,
    ], dim=1).abs()


#@torch.jit.script
def tet_dihedral_angles(
    verts: torch.Tensor,
    tets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute dihedral angles for each tet vertex.

    Based on the method from http://math.stackexchange.com/a/49340/35376

    Args:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.

    Returns:
        d10: (E,) tensor of dihedral angles for the edge between vert 0 and 1.
        d20: (E,) tensor of dihedral angles for the edge between vert 0 and 2.
        d30: (E,) tensor of dihedral angles for the edge between vert 0 and 3.
        d21: (E,) tensor of dihedral angles for the edge between vert 1 and 2.
        d31: (E,) tensor of dihedral angles for the edge between vert 1 and 3.
        d32: (E,) tensor of dihedral angles for the edge between vert 2 and 3.
    """
    eps = 1e-12  # Small constant to avoid division by zero

    # Extract vertices and edges
    v0, v1, v2, v3 = verts[tets].unbind(dim=1)  # Each is (T, 4)
    e10 = torch.sum((v1 - v0)**2, dim=1).sqrt()
    e20 = torch.sum((v2 - v0)**2, dim=1).sqrt()
    e30 = torch.sum((v3 - v0)**2, dim=1).sqrt()
    e21 = torch.sum((v2 - v1)**2, dim=1).sqrt()
    e31 = torch.sum((v3 - v1)**2, dim=1).sqrt()
    e32 = torch.sum((v3 - v2)**2, dim=1).sqrt()

    # Compute triangle face areas w, x, y, and z with Heron's formula
    s = 0.5 * (e20 + e32 + e30)
    w2 = torch.clamp_(s * (s - e20) * (s - e32) * (s - e30), min=0)
    w = torch.sqrt(w2)

    s = 0.5 * (e31 + e21 + e32)
    x2 = torch.clamp_(s * (s - e31) * (s - e21) * (s - e32), min=0)
    x = torch.sqrt(x2)

    s = 0.5 * (e10 + e21 + e20)
    y2 = torch.clamp_(s * (s - e10) * (s - e21) * (s - e20), min=0)
    y = torch.sqrt(y2)

    s = 0.5 * (e10 + e31 + e30)
    z2 = torch.clamp_(s * (s - e10) * (s - e31) * (s - e30), min=0)
    z = torch.sqrt(z2)

    # Compute square of pseudo-faces h, j, k
    # 4 a**2 d**2 - ((b**2 + e**2) - (c**2 + f**2))
    h2 = (4 * e10**2 * e32**2 - ((e20**2 + e31**2) - (e30**2 + e21**2))**2) / 16
    j2 = (4 * e20**2 * e31**2 - ((e30**2 + e21**2) - (e10**2 + e32**2))**2) / 16
    k2 = (4 * e30**2 * e21**2 - ((e10**2 + e32**2) - (e20**2 + e31**2))**2) / 16

    # Compute dihedral angles
    da10 = torch.clamp_((y2 + z2 - h2) / (2 * y * z + eps), min=-1, max=1).acos_()
    da20 = torch.clamp_((z2 + x2 - j2) / (2 * z * x + eps), min=-1, max=1).acos_()
    da30 = torch.clamp_((x2 + y2 - k2) / (2 * x * y + eps), min=-1, max=1).acos_()
    da21 = torch.clamp_((w2 + z2 - k2) / (2 * w * z + eps), min=-1, max=1).acos_()
    da31 = torch.clamp_((w2 + y2 - j2) / (2 * w * y + eps), min=-1, max=1).acos_()
    da32 = torch.clamp_((w2 + x2 - h2) / (2 * w * x + eps), min=-1, max=1).acos_()

    return da10, da20, da30, da21, da31, da32


def laplacian(
    verts: torch.Tensor,
    edges: torch.Tensor,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Returns the uniform non-normalized Laplacian matrix.

    L[i, j] = deg(vi) if i == j else -1

    Args:
        verts: tensor of vertex coordindates of shape (V, 4).
        edges: tensor of edge vertex indices of shape (E, 2).
        dtype: dtype of output.

    Returns:
        L: sparse tensor of shape (V, V).
    """
    nv = len(verts)
    device = verts.device

    idx = torch.cat([edges, edges.fliplr()], dim=0).t()
    ones = torch.ones(1, device=device, dtype=dtype).expand(idx.shape[1])
    deg = torch.zeros(nv, device=device, dtype=dtype)
    deg.scatter_add_(0, idx[0], ones)

    idx_diag = torch.arange(nv)
    idx_diag = torch.stack([idx_diag, idx_diag])

    idx = torch.cat([idx, idx_diag], dim=1)
    values = torch.cat([-ones, deg])

    L = torch.sparse_coo_tensor(idx, values, (nv, nv))

    return L.coalesce()


@torch.jit.script
def norm_laplacian(
    verts: torch.Tensor,
    edges: torch.Tensor,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Returns the uniform normalized Laplacian matrix.

    L[i, j] = 1 if i == j else -1/deg(vi)

    Args:
        verts: tensor of vertex coordindates of shape (V, 4).
        edges: tensor of edge vertex indices of shape (E, 2).
        dtype: dtype of output.

    Returns:
        L: sparse tensor of shape (V, V).
    """
    nv = len(verts)
    device = verts.device

    idx = torch.cat([edges, edges.fliplr()], dim=0).t()
    ones = torch.ones(1, device=device, dtype=dtype).expand(idx.shape[1])
    deg = torch.zeros(nv, device=device, dtype=dtype)
    deg.scatter_add_(0, idx[0], ones)

    zero = torch.zeros(1, device=device, dtype=dtype)
    minus_deg_inv = torch.where(deg > 0.0, -1.0 / deg, zero)
    values = minus_deg_inv[idx[0]]

    idx_diag = torch.arange(nv)
    idx_diag = torch.stack([idx_diag, idx_diag])
    values_diag = torch.ones(nv, device=device, dtype=dtype)

    idx = torch.cat([idx, idx_diag], dim=1)
    values = torch.cat([values, values_diag])

    L = torch.sparse_coo_tensor(idx, values, (nv, nv))

    return L.coalesce()


@torch.jit.script
def scale_dep_laplacian(
    verts: torch.Tensor,
    edges: torch.Tensor,
    edge_lengths: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Returns the scale-dependant Laplacian matrix.

    Computes the laplacian at vertex x_i as:

        L(x_i) = 1/E * sum_{j in N(i)} (x_j - x_i) / | e_ij | ,

    where E = sum_{j in N(i)} |e_ij| and |e_ij| is the length of the edge
    between vertex i and j.

    Note, if all edge lengths are 1, this Laplacian equals norm_laplacian.

    Args:
        verts: tensor of vertex coordindates of shape (V, 4).
        edges: tensor of edge vertex indices of shape (E, 2).
        edge_lengths: optional tensor of edge lengths of length E.
        dtype: dtype of output.

    Returns:
        L: sparse tensor of shape (V, V).
    """
    nv = len(verts)
    device = verts.device

    if edge_lengths is None:
        v0, v1 = verts[edges].unbind(dim=1)
        edge_lengths = torch.sum((v1 - v0)**2, dim=-1).sqrt()  # (E,)

    edge_lengths_rep = edge_lengths.repeat(2)  # (2E,)

    # Compute normalization constant E
    idx = torch.cat([edges, edges.fliplr()], dim=0).t()  # (2, 2E)
    e = torch.zeros(nv, device=device, dtype=dtype)  # (V,)
    e.scatter_add_(0, idx[0], edge_lengths_rep)

    # Compute off-diagonal values
    zero = torch.zeros(1, device=device, dtype=dtype)
    minus_e_inv = torch.where(e > 0.0, -1.0 / e, zero)  # (V,)
    values = minus_e_inv[idx[0]] / edge_lengths_rep  # (2E,)

    # Compute diagonal values
    idx_diag = torch.arange(nv)
    idx_diag = torch.stack([idx_diag, idx_diag])
    sum_inv_lengths = torch.zeros_like(e)
    sum_inv_lengths.scatter_add_(0, idx[0], 1.0 / edge_lengths_rep)
    values_diag = -minus_e_inv * sum_inv_lengths

    # Assemble Laplacian matrix
    idx = torch.cat([idx, idx_diag], dim=1)
    values = torch.cat([values, values_diag])

    L = torch.sparse_coo_tensor(idx, values, (nv, nv))

    return L.coalesce()


def symm_norm_laplacian(
    verts: torch.Tensor,
    edges: torch.Tensor,
    dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Returns the uniform symmetric normalized Laplacian matrix.

    L[i, j] = 1 if i == j else -1/sqrt(deg(vi)*deg(vj))

    Args:
        verts: tensor of vertex coordindates of shape (V, 4).
        edges: tensor of edge vertex indices of shape (E, 2).
        dtype: dtype of output.

    Returns:
        L: sparse tensor of shape (V, V).
    """
    nv = len(verts)
    device = verts.device

    idx = torch.cat([edges, edges.fliplr()], dim=0).t()
    ones = torch.ones(1, device=device, dtype=dtype).expand(idx.shape[1])
    deg = torch.zeros(nv, device=device, dtype=dtype)
    deg.scatter_add_(0, idx[0], ones)

    zero = torch.zeros(1, device=device, dtype=dtype)
    deg_inv_sqrt = torch.where(deg > 0.0, 1.0 / deg.sqrt(), zero)
    values = -1.0 * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]

    idx_diag = torch.arange(nv)
    idx_diag = torch.stack([idx_diag, idx_diag])
    values_diag = torch.ones(nv, device=device, dtype=dtype)

    idx = torch.cat([idx, idx_diag], dim=1)
    values = torch.cat([values, values_diag])

    L = torch.sparse_coo_tensor(idx, values, (nv, nv))

    return L.coalesce()


def fill_boundary(
    verts: torch.Tensor,
    tris: torch.Tensor,
    p: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fill triangular boundary with tets by connecting faces to a new point.

    Args:
        verts: (V, 4) tensor of vertex coordinates. May include non-boundary
            vertices.
        tris: (F, 3) tensor of boundary triangle face vertex indices.
        p: Optional (4,) tensor with new point. Default is mean of boundary
            vertices.

    Returns:
        tets: (T, 4) tensor of tet vertex indices. New point will have use
            len(verts) as index.
        p: (4,) tensor with new point. Same as p if provided.
    """
    if p is None:
        p = verts[tris.unique().long()].mean(dim=0)

    idx = len(verts)
    tets = torch.cat([tris, torch.full((len(tris), 1), idx)], dim=1)
    return tets, p


def orient_tets_away_from(
    verts: torch.Tensor,
    tets: torch.Tensor,
    p: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    centers: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """"
    Permute tets so tet normals point away from point p.

    Args:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.
        p: (3,) tensor with point to orient away from.
        normals: Optional (T, 4) tensor of tet normals.
        centers: Optional (T, 4) tensor of tet center coordinates.

    Returns
        tets_oriented: (T, 4) tensor of permuted tet indices.
    """
    p = torch.as_tensor(p)
    if normals is None:
        normals = tet_normals(verts, tets, normalize=False)
    if centers is None:
        centers = tet_centers(verts, tets)

    tets_oriented = tets.clone()
    # If the dot product between the tet normal and (center - p) is positive
    # the tet is oriented towards p and must be flipped
    flip_idxs = torch.sum((centers - p) * normals, dim=1) < 0.0
    tets_oriented[flip_idxs, 3], tets_oriented[flip_idxs, 2] = \
        tets_oriented[flip_idxs, 2], tets_oriented[flip_idxs, 3]
    return tets_oriented


def make_600_cell(
    center: Optional[torch.Tensor] = None,
    rs: Optional[torch.Tensor] = None,
    rt: Optional[torch.Tensor] = None,
    num_subdiv: int = 0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a 600-cell with a given center and radius.

    The 600-cell is 4D analogue of an icosahedron and is useful for
    approximating a 4D sphere. See: https://en.wikipedia.org/wiki/600-cell

    Args:
        center: Optional center. Default is (0, 0, 0, 0).
        rs: Optional spatial (xyz) radius. Default is 1.
        rt: Optional time (t) raidius. Default is rs.
        num_subdiv: Number of subdivisions.
        device: Torch device to construct 600-cell on.

    Returns:
        verts: (V, 4) tensor of vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.
    """
    if center is None:
        center = torch.zeros(4, device=device)
    if rs is None:
        rs = 1.0
    if rt is None:
        rt = rs
    verts = []

    verts.append(np.array(list(product([-0.5, 0.5], repeat=4))))

    verts.append(np.eye(4))
    verts.append(-np.eye(4))

    perms = np.array(list(permutations(range(4))))
    perm_signs = np.array([np.linalg.det(np.eye(4)[p]) for p in perms])
    even_perms = perms[perm_signs > 0]

    phi = 0.5 * (1.0 + np.sqrt(5.0))  # Golden ratio
    w = 0.5 * np.array([phi, 1.0, 1.0 / phi, 0.0])
    signs = np.array(list(product([-1.0, 1.0], repeat=3)))
    signs = np.concatenate([signs, np.ones((len(signs), 1))], axis=1)
    for p in even_perms:
        verts.append(signs[:, p] * w[p])

    verts = np.concatenate(verts)

    hull = ConvexHull(verts)
    verts = torch.from_numpy(verts)
    tets = torch.from_numpy(hull.simplices).long()

    for i in range(num_subdiv):
        e0, e1 = tet_edges(tets).unbind(1)
        new_verts = 0.5 * (verts[e0] + verts[e1])
        new_verts /= new_verts.norm(dim=1, keepdim=True)
        verts = torch.cat([verts, new_verts], dim=0)

        hull = ConvexHull(verts.numpy())
        tets = torch.from_numpy(hull.simplices).long()

    verts = verts.to(device=device, dtype=torch.float32)
    tets = tets.to(device=device)

    tets = orient_tets_away_from(verts, tets, torch.zeros(4))

    verts[:, :3] *= rs
    verts[:, 3] *= rt
    verts += center

    return verts, tets


def make_4d_disk(
    tri_verts: torch.Tensor,
    tri_faces: torch.Tensor,
    t: float = 0,
    orient_negative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates an oriented 4D disk by tetrahedralizing a triangle mesh.

    Args:
        tri_verts: (Vi, 3) tensor of triangle vertex coordinates.
        tri_faces: (F, 3) tensor of triangle face vertex indices.
        t: Time coordinate for disk. Default is 0.
        orient_negative: Whether to orient the tets towards [0, 0, 0, -inf] or
            (if False) towards [0, 0, 0, +inf].

    Returns:
        verts: (Vo, 4) tensor of tet vertex coordinates.
        tets: (T, 4) tensor of tet vertex indices.
    """
    tri_verts = tri_verts.numpy()
    tri_faces = tri_faces.numpy()
    tri_areas = 0.5 * igl.doublearea(tri_verts, tri_faces)
    tetvol = 0.01 * tri_areas.min() ** 1.5
    tg = tetgen.TetGen(tri_verts, tri_faces)
    verts, tets = tg.tetrahedralize(
        quality=1.1,
        nobisect=True,
        maxvolume=tetvol,
        maxvolume_length=tetvol,
    )
    verts = torch.from_numpy(verts).float()
    tets = torch.from_numpy(tets).long()

    verts = torch.cat([verts, torch.full((len(verts), 1), t)], dim=1)
    p = torch.tensor([0, 0, 0, -1.0 if orient_negative else 1.0]) * 1e3
    tets = orient_tets_away_from(verts, tets, p)

    return verts, tets

