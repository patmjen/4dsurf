#include <vector>
#include <array>
#include <thread>
#include <stdexcept>

#include <torch/extension.h>

#define CHECK_2D(x) TORCH_CHECK((x).dim() == 2, #x " must be 2D but was ", (x).dim(), "D")
#define CHECK_DIM(x, d, v) TORCH_CHECK((x).size(d) == (v), \
                                       #x " must have size ", v, " in dim ", d, " but was ", (x).size(d))
#define CHECK_CPU(x) TORCH_CHECK((x).is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_DTYPE(x, dtype) TORCH_CHECK((x).dtype() == (dtype), #x " must have dtype " #dtype)
#define CHECK_NUMEL(x, n) TORCH_CHECK((x).numel() == (n), \
                                      #x " must have ", n, " elements but had ", (x).numel())

constexpr double eps = 1e-12;

using dvec4 = std::array<double, 4>;

dvec4 operator+(const dvec4& x, const dvec4& y)
{
    return { x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3] };
}

dvec4 operator-(const dvec4& x, const dvec4& y)
{
    return { x[0] - y[0], x[1] - y[1], x[2] - y[2], x[3] - y[3] };
}

dvec4 operator*(double a, const dvec4& x)
{
    return { a * x[0], a * x[1], a * x[2], a * x[3] };
}

dvec4 operator*(const dvec4& x, double a)
{
    return a * x;
}

double norm(const dvec4& x)
{
    return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]);
}

double norm_sqrd(const dvec4& x)
{
    return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
}

/*double sqrd_pseudo_face(double a, double b, double c, double d, double e, double f)
{
    constexpr double div16 = 1.0 / 16.0;
    const double s = (b*b + e*e) - (c*c + f*f);
    return (4 * a*a * d*d - s*s)  * div16;
}*/
double sqrd_pseudo_face(double a2, double b2, double c2, double d2, double e2, double f2)
{
    constexpr double div16 = 1.0 / 16.0;
    const double s = (b2 + e2) - (c2 + f2);
    return (4.0 * a2 * d2 - s*s)  * div16;
}

double clamp_unit(double x)
{
    return fmax(-1.0, fmin(1.0, x));
}

std::vector<torch::Tensor> tet_dihedral_angles(
    torch::Tensor verts,
    torch::Tensor tets)
{
    // Validate input
    CHECK_CPU(verts);
    CHECK_CONTIGUOUS(verts);
    CHECK_2D(verts);
    CHECK_DIM(verts, 1, 4);
    CHECK_DTYPE(verts, torch::kFloat64);

    CHECK_CPU(tets);
    CHECK_CONTIGUOUS(tets);
    CHECK_2D(tets);
    CHECK_DIM(tets, 1, 4);
    CHECK_DTYPE(verts, torch::kInt32);

    auto num_tets = tets.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto da10 = torch::empty({ num_tets }, options);
    auto da20 = torch::empty({ num_tets }, options);
    auto da30 = torch::empty({ num_tets }, options);
    auto da21 = torch::empty({ num_tets }, options);
    auto da31 = torch::empty({ num_tets }, options);
    auto da32 = torch::empty({ num_tets }, options);

    auto verts_a = verts.accessor<double, 2>();
    auto tets_a = tets.accessor<int64_t, 2>();
    auto da10_a = da10.accessor<double, 1>();
    auto da20_a = da20.accessor<double, 1>();
    auto da30_a = da30.accessor<double, 1>();
    auto da21_a = da21.accessor<double, 1>();
    auto da31_a = da31.accessor<double, 1>();
    auto da32_a = da32.accessor<double, 1>();

    std::vector<std::thread> threads;
    int num_threads = std::thread::hardware_concurrency();
    int tets_per_tread = num_tets / num_threads + (num_tets % num_threads == 0 ? 0 : 1);
    if (num_tets <= 10 * num_threads) {
        num_threads = 1;
        tets_per_tread = num_tets;
    }
    const size_t tets_per_thread = num_tets / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        size_t begin = i * tets_per_tread;
        size_t end = std::min<size_t>(begin + tets_per_tread, num_tets);
        threads.emplace_back([&](size_t begin, size_t end) {
            for (size_t t = begin; t < end; ++t) {
                auto tet = tets_a[t];
                int64_t vk0 = tet[0];
                int64_t vk1 = tet[1];
                int64_t vk2 = tet[2];
                int64_t vk3 = tet[3];
                const dvec4 v0 = { verts_a[vk0][0], verts_a[vk0][1], verts_a[vk0][2], verts_a[vk0][3] };
                const dvec4 v1 = { verts_a[vk1][0], verts_a[vk1][1], verts_a[vk1][2], verts_a[vk1][3] };
                const dvec4 v2 = { verts_a[vk2][0], verts_a[vk2][1], verts_a[vk2][2], verts_a[vk2][3] };
                const dvec4 v3 = { verts_a[vk3][0], verts_a[vk3][1], verts_a[vk3][2], verts_a[vk3][3] };

                const double e10_2 = norm_sqrd(v1 - v0);
                const double e20_2 = norm_sqrd(v2 - v0);
                const double e30_2 = norm_sqrd(v3 - v0);
                const double e21_2 = norm_sqrd(v2 - v1);
                const double e31_2 = norm_sqrd(v3 - v1);
                const double e32_2 = norm_sqrd(v3 - v2);

                const double e10 = sqrt(e10_2);
                const double e20 = sqrt(e20_2);
                const double e30 = sqrt(e30_2);
                const double e21 = sqrt(e21_2);
                const double e31 = sqrt(e31_2);
                const double e32 = sqrt(e32_2);

                double s = 0.5 * (e20 + e32 + e30);
                const double w2 = fmax(0.0, s * (s - e20) * (s - e32) * (s - e30));
                const double w = sqrt(w2);

                s = 0.5 * (e31 + e21 + e32);
                double x2 = fmax(0.0, s * (s - e31) * (s - e21) * (s - e32));
                double x = sqrt(x2);

                s = 0.5 * (e10 + e21 + e20);
                const double y2 = fmax(0.0, s * (s - e10) * (s - e21) * (s - e20));
                const double y = sqrt(y2);

                s = 0.5 * (e10 + e31 + e30);
                const double z2 = fmax(0.0, s * (s - e10) * (s - e31) * (s - e30));
                const double z = sqrt(z2);

                const double h2 = sqrd_pseudo_face(e10_2, e20_2, e30_2, e32_2, e31_2, e21_2);
                const double j2 = sqrd_pseudo_face(e20_2, e30_2, e10_2, e31_2, e21_2, e32_2);
                const double k2 = sqrd_pseudo_face(e30_2, e10_2, e20_2, e21_2, e32_2, e31_2);

                da10_a[t] = acos(clamp_unit((y2 + z2 - h2) / (2 * y * z + eps)));
                da20_a[t] = acos(clamp_unit((z2 + x2 - j2) / (2 * z * x + eps)));
                da30_a[t] = acos(clamp_unit((x2 + y2 - k2) / (2 * x * y + eps)));
                da21_a[t] = acos(clamp_unit((w2 + z2 - k2) / (2 * w * z + eps)));
                da31_a[t] = acos(clamp_unit((w2 + y2 - j2) / (2 * w * y + eps)));
                da32_a[t] = acos(clamp_unit((w2 + x2 - h2) / (2 * w * x + eps)));
            }
        }, begin, end);
    }

    for (auto &th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    return { da10, da20, da30, da21, da31, da32 };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tet_dihedral_angles", &tet_dihedral_angles, "Compute tet. dihedral angles.");
}