#include <array>
#include <cinttypes>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

#include <torch/extension.h>

using namespace torch::indexing;

#define CHECK_2D(x) TORCH_CHECK((x).dim() == 2, #x " must be 2D but was ", (x).dim(), "D")
#define CHECK_DIM(x, d, v) TORCH_CHECK((x).size(d) == (v), \
                                       #x " must have size ", v, " in dim ", d, " but was ", (x).size(d))
#define CHECK_CPU(x) TORCH_CHECK((x).is_cpu(), #x " must be a CPU tensor")
#define CHECK_NUMEL(x, n) TORCH_CHECK((x).numel() == (n), \
                                      #x " must have ", n, " elements but had ", (x).numel())

constexpr float TOL = 100 * std::numeric_limits<float>::epsilon();
constexpr int64_t MAX_VERTS = std::numeric_limits<uint32_t>::max();

std::array<float, 4> cross4(const std::array<float, 4>& t,
                            const std::array<float, 4>& u,
                            const std::array<float, 4>& v)
{
	// See: https://math.stackexchange.com/a/2371039
	float a1 = t[3] * u[2] * v[1] - t[2] * u[3] * v[1] - t[3] * u[1] * v[2] +
		t[1] * u[3] * v[2] + t[2] * u[1] * v[3] - t[1] * u[2] * v[3];
	float a2 = -t[3] * u[2] * v[0] + t[2] * u[3] * v[0] + t[3] * u[0] * v[2] -
		t[0] * u[3] * v[2] - t[2] * u[0] * v[3] + t[0] * u[2] * v[3];
	float a3 = t[3] * u[1] * v[0] - t[1] * u[3] * v[0] - t[3] * u[0] * v[1] +
		t[0] * u[3] * v[1] + t[1] * u[0] * v[3] - t[0] * u[1] * v[3];
	float a4 = -t[2] * u[1] * v[0] + t[1] * u[2] * v[0] + t[2] * u[0] * v[1] -
		t[0] * u[2] * v[1] - t[1] * u[0] * v[2] + t[0] * u[1] * v[2];
	return std::array<float, 4>({ a1, a2, a3, a4 });
}

float dot(const std::array<float, 4>& x, const std::array<float, 4>& y)
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

std::array<float, 4> operator+(const std::array<float, 4>& x, const std::array<float, 4>& y)
{
    return { x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3] };
}

std::array<float, 4> operator-(const std::array<float, 4>& x, const std::array<float, 4>& y)
{
    return { x[0] - y[0], x[1] - y[1], x[2] - y[2], x[3] - y[3] };
}

std::array<float, 4> operator*(float a, const std::array<float, 4>& x)
{
    return { a * x[0], a * x[1], a * x[2], a * x[3] };
}

std::array<float, 4> operator*(const std::array<float, 4>& x, float a)
{
    return a * x;
}

inline int64_t make_key(int64_t vk1, int64_t vk2)
{
    if (vk1 < vk2) {
        return (static_cast<int64_t>(vk1) << 32) ^ static_cast<int64_t>(vk2);
    } else {
        return (static_cast<int64_t>(vk2) << 32) ^ static_cast<int64_t>(vk1);
    }
}

std::array<int64_t, 4> arg_sort_4(int64_t v1, int64_t v2, int64_t v3, int64_t v4)
{
    std::array<int64_t, 4> vals = { v1, v2, v3, v4 };
    std::array<int64_t, 4> idxs = { 0, 1, 2, 3 };
    // Just use bubble sort, since we need so few iterations
    for (int64_t si = 0; si < 3; ++si) {
        for (int64_t i = 0; i < 4 - 1 - si; ++i) {
            if (vals[i] > vals[i + 1]) {
                std::swap(vals[i], vals[i + 1]);
                std::swap(idxs[i], idxs[i + 1]);
            }
        }
    }
    return idxs;
}

std::vector<torch::Tensor> hyperplane_intersection(
    torch::Tensor verts,
    torch::Tensor tets,
    torch::Tensor plane_normal_tensor,
    torch::Tensor plane_dist)
{
    // Validate input
    CHECK_CPU(verts);
    CHECK_2D(verts);
    CHECK_DIM(verts, 1, 4);
    TORCH_CHECK(verts.size(0) < MAX_VERTS,\
                "Cannot handle vertex count over ", MAX_VERTS, \
                " (was: ", verts.size(0), ")");

    CHECK_CPU(tets);
    CHECK_2D(tets);
    CHECK_DIM(tets, 1, 4);

    auto verts_a = verts.accessor<float, 2>();
    auto tets_a = tets.accessor<int64_t, 2>();
    auto num_tets = tets.size(0);

    auto plane_normal_a = plane_normal_tensor.accessor<float, 1>();
    std::array<float, 4> plane_normal = {
        plane_normal_a[0],
        plane_normal_a[1],
        plane_normal_a[2],
        plane_normal_a[3],
    };

    std::unordered_map<int64_t, int64_t> edge_to_point;
    std::vector<int64_t> int_tet_idxs;
    std::vector<int64_t> tri_face_to_tet;
    std::vector<std::array<int64_t, 3>> int_tris;
    std::vector<std::array<float, 4>> int_tri_verts;

    auto add_vertex = [&](int64_t vk, const std::array<float, 4>& v)
    {
        auto key = make_key(vk, vk);
        if (edge_to_point.find(key) == edge_to_point.end()) {
            // Key not seen before, so add the new vertex
            int_tri_verts.push_back(v);
            edge_to_point[key] = int_tri_verts.size() - 1;
        }
        return edge_to_point[key];
    };

    auto add_intersection = [&](int64_t vk1, int64_t vk2, const std::array<float, 4>& v1, const std::array<float, 4>& v2,
                                int64_t s1, int64_t s2, float d1, float d2)
    {
        assert(s1 != s2); // Both vertices may **not** be on the same side

        if (s1 == 0) {
            // First vertex is in hyperplane
            return add_vertex(vk1, v1);
        } else if (s2 == 0) {
            // Second vertex is in hyperplane
            return add_vertex(vk2, v2);
        } else {
            // No vertex is in the hyperplane, so compute intersection
            auto key = make_key(vk1, vk2);
            if (edge_to_point.find(key) == edge_to_point.end()) {
                // Pair not seen before, so add the new vertex
                float a = d1 / (d1 - d2);
                std::array<float, 4> p = (1.0 - a) * v1 + a * v2;
                int_tri_verts.push_back(p);
                edge_to_point[key] = int_tri_verts.size() - 1;
            }
            return edge_to_point[key];
        }
    };

    // Compute vertex-to-plane distances and signs with a tolerance for robustness
    auto vert_dots = torch::sum(verts * plane_normal_tensor, 1) - plane_dist;
    vert_dots.index_put_({ vert_dots.abs() < TOL }, 0.0);
    auto vert_dots_a = vert_dots.accessor<float, 1>();

    auto vert_signs = torch::sign(vert_dots).toType(torch::kLong);
    auto vert_signs_a = vert_signs.accessor<int64_t, 1>();

    for (int64_t ti = 0; ti < num_tets; ++ti) {
        auto tet = tets_a[ti];
        int64_t vk0 = tet[0];
        int64_t vk1 = tet[1];
        int64_t vk2 = tet[2];
        int64_t vk3 = tet[3];
        const std::array<float, 4> v0 = { verts_a[vk0][0], verts_a[vk0][1], verts_a[vk0][2], verts_a[vk0][3] };
        const std::array<float, 4> v1 = { verts_a[vk1][0], verts_a[vk1][1], verts_a[vk1][2], verts_a[vk1][3] };
        const std::array<float, 4> v2 = { verts_a[vk2][0], verts_a[vk2][1], verts_a[vk2][2], verts_a[vk2][3] };
        const std::array<float, 4> v3 = { verts_a[vk3][0], verts_a[vk3][1], verts_a[vk3][2], verts_a[vk3][3] };

        int64_t s0 = vert_signs_a[vk0];
        int64_t s1 = vert_signs_a[vk1];
        int64_t s2 = vert_signs_a[vk2];
        int64_t s3 = vert_signs_a[vk3];

        int64_t npos = (s0 > 0) + (s1 > 0) + (s2 > 0) + (s3 > 0);
        int64_t nneg = (s0 < 0) + (s1 < 0) + (s2 < 0) + (s3 < 0);
        int64_t nzer = 4 - npos - nneg;

        if (npos == 4 || nneg == 4) {
            // No intersection, move on to next
            continue;
        }
        int_tet_idxs.push_back(ti); // Mark tet as intersecting
        if (nzer == 4) {
            // Intersection is whole tet
            int64_t nvk0 = add_vertex(vk0, v0);
            int64_t nvk1 = add_vertex(vk1, v1);
            int64_t nvk2 = add_vertex(vk2, v2);
            int64_t nvk3 = add_vertex(vk3, v3);
            int_tris.push_back({ nvk0, nvk1, nvk2 });
            int_tris.push_back({ nvk0, nvk1, nvk3 });
            int_tris.push_back({ nvk0, nvk2, nvk3 });
            int_tris.push_back({ nvk1, nvk2, nvk3 });

            tri_face_to_tet.push_back(ti);
            tri_face_to_tet.push_back(ti);
            tri_face_to_tet.push_back(ti);
            tri_face_to_tet.push_back(ti);
        } else if (npos == 2 && nneg == 2) {
            // Intersection is a quad
            const auto idxs = arg_sort_4(s0, s1, s2, s3);

            const std::array<int64_t, 4> tetKeys = { vk0, vk1, vk2, vk3 };
            const std::array<std::array<float, 4>, 4> tetVerts = { v0, v1, v2, v3 };
            const std::array<int64_t, 4> tetSigns = { s0, s1, s2, s3 };
            const std::array<float, 4> tetDists = {
                vert_dots_a[vk0],
                vert_dots_a[vk1],
                vert_dots_a[vk2],
                vert_dots_a[vk3]
            };
            auto add_tet_intersection = [&](int64_t i1, int64_t i2)
            {
                return add_intersection(
                    tetKeys[i1], tetKeys[i2],
                    tetVerts[i1], tetVerts[i2],
                    tetSigns[i1], tetSigns[i2],
                    tetDists[i1], tetDists[i2]
                );
            };

            int64_t nvk0 = add_tet_intersection(idxs[0], idxs[2]);
            int64_t nvk1 = add_tet_intersection(idxs[0], idxs[3]);
            int64_t nvk2 = add_tet_intersection(idxs[1], idxs[2]);
            int64_t nvk3 = add_tet_intersection(idxs[1], idxs[3]);

            int_tris.push_back({ nvk0, nvk1, nvk2 });
            int_tris.push_back({ nvk3, nvk1, nvk2 });

            tri_face_to_tet.push_back(ti);
            tri_face_to_tet.push_back(ti);
        } else if (nzer == 3) {
            // Intersection is a triangular tet face
            const auto idxs = arg_sort_4(abs(s0), abs(s1), abs(s2), abs(s3)); // We don't care about above or below
            const std::array<int64_t, 4> tetKeys = { vk0, vk1, vk2, vk3 };
            const std::array<std::array<float, 4>, 4> tetVerts = { v0, v1, v2, v3 };
            std::array<int64_t, 3> newKeys;
            for (int64_t i = 0; i < 3; ++i) {
                newKeys[i] = add_vertex(tetKeys[idxs[i]], tetVerts[idxs[i]]);
            }
            int_tris.push_back({ newKeys[0], newKeys[1], newKeys[2] });

            tri_face_to_tet.push_back(ti);
        }else if (npos == 1 || nneg == 1) {
            // Intersection is a triangle
            const auto idxs = arg_sort_4(s0, s1, s2, s3);
            int64_t idxA, idxB1, idxB2, idxB3;
            if (npos == 1) {
                idxA = idxs[3];
                idxB1 = idxs[0];
                idxB2 = idxs[1];
                idxB3 = idxs[2];
            } else {
                idxA = idxs[0];
                idxB1 = idxs[1];
                idxB2 = idxs[2];
                idxB3 = idxs[3];
            }

            const std::array<int64_t, 4> tetKeys = { vk0, vk1, vk2, vk3 };
            const std::array<std::array<float, 4>, 4> tetVerts = { v0, v1, v2, v3 };
            const std::array<int64_t, 4> tetSigns = { s0, s1, s2, s3 };
            const std::array<float, 4> tetDists = {
                vert_dots_a[vk0],
                vert_dots_a[vk1],
                vert_dots_a[vk2],
                vert_dots_a[vk3]
            };
            auto addTetIntersection = [&](int64_t i1, int64_t i2)
            {
                return add_intersection(
                    tetKeys[i1], tetKeys[i2],
                    tetVerts[i1], tetVerts[i2],
                    tetSigns[i1], tetSigns[i2],
                    tetDists[i1], tetDists[i2]
                );
            };

            int64_t nvk0 = addTetIntersection(idxA, idxB1);
            int64_t nvk1 = addTetIntersection(idxA, idxB2);
            int64_t nvk2 = addTetIntersection(idxA, idxB3);

            int_tris.push_back({ nvk0, nvk1, nvk2 });

            tri_face_to_tet.push_back(ti);
        } else if (nzer == 2 && (npos == 2 || nneg == 2)) {
            // Intersection is line segment
            /*
            const auto idxs = arg_sort_4(abs(s0), abs(s1), abs(s2), abs(s3)); // We don't care about above or below

            const std::array<int64_t, 4> tetKeys = { vk0, vk1, vk2, vk3 };
            const std::array<std::array<float, 4>, 4> tetVerts = { v0, v1, v2, v3 };

            int64_t nvk0 = add_vertex(tetKeys[idxs[0]], tetVerts[idxs[0]]);
            int64_t nvk1 = add_vertex(tetKeys[idxs[1]], tetVerts[idxs[1]]);

            int_tris.push_back({ nvk0, nvk1, -1 });

            tri_face_to_tet.push_back(ti);
            */
        } else if (nzer == 1 && (npos == 3 || nneg == 3)) {
            // Intersection is a point
            const auto idxs = arg_sort_4(abs(s0), abs(s1), abs(s2), abs(s3)); // We don't care about above or below

            const std::array<int64_t, 4> tetKeys = { vk0, vk1, vk2, vk3 };
            const std::array<std::array<float, 4>, 4> tetVerts = { v0, v1, v2, v3 };

            add_vertex(tetKeys[idxs[0]], tetVerts[idxs[0]]);
        } else {
            TORCH_CHECK(false, "Invalid tet. configuration"); // We should never hit this!
        }
    }

    auto int_tri_verts_tensor = torch::empty({ (int64_t)int_tri_verts.size(), 4 });
    auto int_tri_verts_a = int_tri_verts_tensor.accessor<float, 2>();
    for (int64_t i = 0; i < int_tri_verts.size(); ++i) {
        int_tri_verts_a[i][0] = int_tri_verts[i][0];
        int_tri_verts_a[i][1] = int_tri_verts[i][1];
        int_tri_verts_a[i][2] = int_tri_verts[i][2];
        int_tri_verts_a[i][3] = int_tri_verts[i][3];
    }

    auto options_int = torch::TensorOptions().dtype(torch::kLong);
    auto int_tris_tensor = torch::empty({ (int64_t)int_tris.size(), 3 }, options_int);
    auto int_tris_a = int_tris_tensor.accessor<int64_t, 2>();
    for (int64_t i = 0; i < int_tris.size(); ++i) {
        int_tris_a[i][0] = int_tris[i][0];
        int_tris_a[i][1] = int_tris[i][1];
        int_tris_a[i][2] = int_tris[i][2];
    }

    return { int_tri_verts_tensor, int_tris_tensor };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hyperplane_intersection", &hyperplane_intersection, "Hyperplane intersection");
}
