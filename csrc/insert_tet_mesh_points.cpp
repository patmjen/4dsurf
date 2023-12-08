#include <vector>
#include <stdexcept>

#include <torch/extension.h>

#include "tetgen.h"

#define CHECK_2D(x) TORCH_CHECK((x).dim() == 2, #x " must be 2D but was ", (x).dim(), "D")
#define CHECK_DIM(x, d, v) TORCH_CHECK((x).size(d) == (v), \
                                       #x " must have size ", v, " in dim ", d, " but was ", (x).size(d))
#define CHECK_CPU(x) TORCH_CHECK((x).is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_DTYPE(x, dtype) TORCH_CHECK((x).dtype() == (dtype), #x " must have dtype " #dtype)
#define CHECK_NUMEL(x, n) TORCH_CHECK((x).numel() == (n), \
                                      #x " must have ", n, " elements but had ", (x).numel())

std::vector<torch::Tensor> insert_tet_mesh_points(
    torch::Tensor verts,
    torch::Tensor tets,
    torch::Tensor add_verts)
{
     // Validate input
    CHECK_CPU(verts);
    CHECK_CONTIGUOUS(verts);
    CHECK_2D(verts);
    CHECK_DIM(verts, 1, 3);
    CHECK_DTYPE(verts, torch::kFloat64);

    CHECK_CPU(tets);
    CHECK_CONTIGUOUS(tets);
    CHECK_2D(tets);
    CHECK_DIM(tets, 1, 4);
    CHECK_DTYPE(verts, torch::kInt32);

    CHECK_CPU(add_verts);
    CHECK_CONTIGUOUS(add_verts);
    CHECK_2D(add_verts);
    CHECK_DIM(add_verts, 1, 3);
    CHECK_DTYPE(verts, torch::kFloat64);

    auto num_verts = verts.size(0);
    auto num_tets = tets.size(0);
    auto num_add_verts = add_verts.size(0);

    tetgenio io_in;
    // Add vertices
    io_in.pointlist = verts.data_ptr<double>(); // verts are stored row-major so they fit TetGen's API
    io_in.numberofpoints = num_verts;

    // Add tets
    io_in.tetrahedronlist = tets.data_ptr<int>(); // tets are stored row-major so they fit TetGen's API
    io_in.numberoftetrahedra = num_tets;

    // Add new vertices
    tetgenio io_add;
    io_add.pointlist = add_verts.data_ptr<double>(); // verts are stored row-major so they fit TetGen's API
    io_add.numberofpoints = num_add_verts;

    tetgenio io_out;
    try {
        tetrahedralize("riQMJY", &io_in, &io_out, &io_add);
    } catch (int err) {
        io_in.pointlist = NULL;
        io_in.tetrahedronlist = NULL;
        io_add.pointlist = NULL;
        switch (err) {
        case 1:
            TORCH_CHECK(false, "tetgen: (1) out of memory");
        case 2:
            TORCH_CHECK(false, "tetgen: (2) internal error");
        case 3:
            TORCH_CHECK(false, "tetgen: (3) input surface mesh contains self intersections");
        case 4:
            TORCH_CHECK(false, "tetgen: (4) very small input feature size was detected");
        case 5:
            TORCH_CHECK(false, "tetgen: (5) two very close input facets detected");
        case 10:
            TORCH_CHECK(false, "tetgen: (10) input error");
        case 200:
            TORCH_CHECK(false, "tetgen: (200) boundary contains Steiner points");
        default:
            TORCH_CHECK(false, "tetgen: (", err, ") unkown error code");
        }
    } catch (...) {
        io_in.pointlist = NULL;
        io_in.tetrahedronlist = NULL;
        io_add.pointlist = NULL;
        TORCH_CHECK(false, "unknown error");
    }

    // Clear tetgenio pointers so they won't free PyTorch memory
    io_in.pointlist = NULL;
    io_in.tetrahedronlist = NULL;
    io_add.pointlist = NULL;

    // Copy output to tensors and return
    auto options_verts = torch::TensorOptions().dtype(torch::kFloat64);
    auto new_verts = torch::empty({ (int64_t)io_out.numberofpoints, 3 }, options_verts);
    auto new_verts_a = new_verts.accessor<double, 2>();
    for (int i = 0; i < io_out.numberofpoints; ++i) {
        new_verts_a[i][0] = io_out.pointlist[i * 3 + 0];
        new_verts_a[i][1] = io_out.pointlist[i * 3 + 1];
        new_verts_a[i][2] = io_out.pointlist[i * 3 + 2];
    }

    auto options_tets = torch::TensorOptions().dtype(torch::kLong);
    auto new_tets = torch::empty({ (int64_t)io_out.numberoftetrahedra, 4 }, options_tets);
    auto new_tets_a = new_tets.accessor<int64_t, 2>();
    for (int i = 0; i < io_out.numberoftetrahedra; ++i) {
        new_tets_a[i][0] = io_out.tetrahedronlist[i * 4 + 0];
        new_tets_a[i][1] = io_out.tetrahedronlist[i * 4 + 1];
        new_tets_a[i][2] = io_out.tetrahedronlist[i * 4 + 2];
        new_tets_a[i][3] = io_out.tetrahedronlist[i * 4 + 3];
    }

    return { new_verts, new_tets };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("insert_tet_mesh_points", &insert_tet_mesh_points, "Insert new points in tet. mesh using TetGen.");
}