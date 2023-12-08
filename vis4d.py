import sys

import numpy as np
import torch
from torch.utils.cpp_extension import load
import polyscope as ps
import polyscope.imgui as psim

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

_C = load('_C', ['csrc/hyperplane_intersection.cpp'],
          extra_cflags=_EXTRA_CFLAGS)


class Structure:
    def __init__(self, structure_type, name, gui):
        self.structure_type = structure_type
        self.structure_name = name
        self.gui = gui


    def add_scalar_quantity(self, name, values, **kwargs):
        self.gui.add_scalar_quantity(
            self.structure_type,
            self.structure_name,
            name,
            values,
            **kwargs,
        )


    def add_color_quantity(self, name, values, **kwargs):
        self.gui.add_color_quantity(
            self.structure_type,
            self.structure_name,
            name,
            values,
            **kwargs,
        )


class Vis4dGui:
    def __init__(self):
        self.ui_normal_options = ['XYZ', 'XYT', 'XZT', 'YZT']
        self.plane_normals = {
            'XYZ': torch.tensor([0.0, 0.0, 0.0, 1.0]),
            'XYT': torch.tensor([0.0, 0.0, 1.0, 0.0]),
            'XZT': torch.tensor([0.0, 1.0, 0.0, 0.0]),
            'YZT': torch.tensor([1.0, 0.0, 0.0, 0.0]),
        }
        self.plane_axes_idxs = {
            'XYZ': [0, 1, 2],
            'XYT': [0, 1, 3],
            'XZT': [0, 2, 3],
            'YZT': [1, 2, 3],
        }
        self.plane_normal_axis_idx = {
            'XYZ': 3,
            'XYT': 2,
            'XZT': 1,
            'YZT': 0,
        }
        self.ui_plane_dists = torch.zeros(4)

        self.ui_normal_selected = self.ui_normal_options[0]

        self.registered_meshes = dict()
        self.registered_curves = dict()
        self.registered_points = dict()

        self.structure_keys = ['surface', 'curve_network', 'point_cloud']
        self.struct_getters = {
            'surface': ps.get_surface_mesh,
            'curve_network': ps.get_curve_network,
            'point_cloud': ps.get_point_cloud,
        }
        self.scalar_quantities = []
        self.color_quantities = []

        self.min_coords = torch.full((4,), np.inf)
        self.max_coords = torch.full((4,), -np.inf)


    def set_plane_axes(self, ax: str):
        ax = ax.upper()
        if ax not in self.ui_normal_options:
            opt_str = ', '.join(self.ui_normal_options).strip()
            raise ValueError(f"ax must be one of {opt_str} but was {ax}")


    def reset_ps_viewer(self):
        ps.clear_user_callback()
        ps.remove_all_structures()
        for _ in range(10):
            ps.remove_last_scene_slice_plane()


    def _update_min_max_coords(self, verts):
        verts_min = verts.min(dim=0).values
        verts_max = verts.max(dim=0).values
        self.min_coords = torch.minimum(self.min_coords, verts_min)
        self.max_coords = torch.maximum(self.min_coords, verts_max)
        self.ui_plane_dists = 0.5 * (self.min_coords + self.max_coords)


    def register_hypersurface_mesh(self, name, verts, tets, **kwargs):
        verts = torch.as_tensor(verts)
        tets = torch.as_tensor(tets)
        self.registered_meshes[name] = (verts, tets, kwargs)
        self._update_min_max_coords(verts)
        return Structure('surface', name, self)


    def register_curve_network(self, name, verts, edges, **kwargs):
        verts = torch.as_tensor(verts)
        edges = torch.as_tensor(edges)
        self.registered_curves[name] = (verts, edges, kwargs)
        self._update_min_max_coords(verts)
        return Structure('curve_network', name, self)


    def register_point_cloud(self, name, verts, **kwargs):
        verts = torch.as_tensor(verts)
        self.registered_points[name] = (verts, kwargs)
        self._update_min_max_coords(verts)
        return Structure('point_cloud', name, self)


    def add_scalar_quantity(self, structure, struct_name, quantity_name,
                            values, **kwargs):
        if structure not in self.structure_keys:
            opt_str = ', '.join(self.structure_keys).strip()
            raise ValueError(
                f"structure must be one of {opt_str} but was {structure}"
            )
        entry = (structure, struct_name, quantity_name, values, kwargs)
        self.scalar_quantities.append(entry)


    def add_color_quantity(self, structure, struct_name, quantity_name,
                            values, **kwargs):
        if structure not in self.structure_keys:
            opt_str = ', '.join(self.structure_keys).strip()
            raise ValueError(
                f"structure must be one of {opt_str} but was {structure}"
            )
        entry = (structure, struct_name, quantity_name, values, kwargs)
        self.color_quantities.append(entry)


    def show(self, reset_ps=True):
        if reset_ps:
            self.reset_ps_viewer()

        ps.set_user_callback(self._ps_ui_callback)
        ps.set_automatically_compute_scene_extents(False)
        self._render()
        ps.show()


    def _ps_ui_callback(self):
        ax_idx = self.plane_normal_axis_idx[self.ui_normal_selected]
        min_coord = self.min_coords[ax_idx]
        max_coord = self.max_coords[ax_idx]
        must_render, new_ui_plane_dist = psim.SliderFloat(
            "Hyperplane dist.",
            self.ui_plane_dists[ax_idx],
            v_min=min_coord,
            v_max=max_coord
        )
        self.ui_plane_dists[ax_idx] = new_ui_plane_dist
        changed_normal = psim.BeginCombo("Hyperplane axes",
                                         self.ui_normal_selected)
        if changed_normal:
            for v in self.ui_normal_options:
                _, selected = psim.Selectable(v, self.ui_normal_selected==v)
                if selected and v != self.ui_normal_selected:
                    self.ui_normal_selected = v
                    must_render = True

        psim.EndCombo()

        if must_render:
            self._render()


    def _render(self):
        plane_normal = self.plane_normals[self.ui_normal_selected]
        plane_axes_idx = self.plane_axes_idxs[self.ui_normal_selected]
        ax_idx = self.plane_normal_axis_idx[self.ui_normal_selected]
        plane_dist = self.ui_plane_dists[ax_idx]
        for name, (verts, tets, kwargs) in self.registered_meshes.items():
            tri_verts, tri_faces = _C.hyperplane_intersection(
                verts, tets, plane_normal, plane_dist
            )
            if 'back_face_policy' not in kwargs:
                kwargs['back_face_policy'] = 'identical'
            ps.register_surface_mesh(
                name,
                vertices=tri_verts[:, plane_axes_idx].numpy(),
                faces=tri_faces.numpy(),
                **kwargs,
            )

        for name, (verts, edges, kwargs) in self.registered_curves.items():
            ps.register_curve_network(
                name,
                nodes=verts[:, plane_axes_idx].numpy(),
                edges=edges.numpy(),
                **kwargs,
            )

        for name, (verts, kwargs) in self.registered_points.items():
            ps.register_point_cloud(
                name,
                points=verts[:, plane_axes_idx].numpy(),
                **kwargs,
            )

        for entry in self.scalar_quantities:
            (structure, struct_name, quantity_name, values, kwargs) = entry
            s = self.struct_getters[structure](struct_name)
            s.add_scalar_quantity(quantity_name, values, **kwargs)

        for entry in self.color_quantities:
            (structure, struct_name, quantity_name, values, kwargs) = entry
            s = self.struct_getters[structure](struct_name)
            s.add_color_quantity(quantity_name, values, **kwargs)
