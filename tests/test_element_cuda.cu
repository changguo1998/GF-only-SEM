/**
 * @file test_element_cuda.cu
 * @brief CUDA element residual tests — compare CUDA result against CPU reference.
 *
 * Builds only when GF_WITH_CUDA is enabled.
 * Each test generates random input, runs both CPU and CUDA backends,
 * and compares the residual with tight tolerance.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "gf/backend.hpp"
#include "gf/element.hpp"
#include "gf/gll.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

namespace {

/// Fill a vector with random uniform values in [-1, 1].
void random_fill(std::vector<double>& v, unsigned seed = 42) {
    std::srand(seed);
    for (auto& x : v) {
        x = 2.0 * static_cast<double>(std::rand()) / RAND_MAX - 1.0;
    }
}

/// Build a random element at polynomial order N.
struct RandomElement {
    int ngll;
    int n_node;
    std::vector<double> dxi_dx;
    std::vector<double> jacobian;
    std::vector<double> vp;
    std::vector<double> vs;
    std::vector<double> density;
    std::vector<double> D;
    std::vector<double> w;
    std::vector<double> nodes;

    RandomElement(int N, unsigned seed = 123) : ngll(N + 1), n_node((N + 1) * (N + 1) * (N + 1)) {
        nodes = gll_nodes(N);
        w = gll_weights(N, nodes);
        D = gll_derivative_matrix(N, nodes);

        dxi_dx.resize(n_node * 9);
        jacobian.resize(n_node);
        vp.resize(n_node);
        vs.resize(n_node);
        density.resize(n_node);

        // Random dxi_dx with positive diagonal (physically plausible)
        random_fill(dxi_dx, seed);
        // Ensure diagonal dominance for invertibility
        for (int n = 0; n < n_node; ++n) {
            dxi_dx[n * 9 + 0] += 3.0;  // dξ/dx
            dxi_dx[n * 9 + 4] += 3.0;  // dη/dy
            dxi_dx[n * 9 + 8] += 3.0;  // dζ/dz
        }

        random_fill(jacobian, seed + 1);
        for (auto& j : jacobian)
            j = std::abs(j) + 0.01;  // ensure positive

        random_fill(vp, seed + 2);
        for (auto& v : vp)
            v = std::abs(v) + 1000.0;  // vp > 1000

        random_fill(vs, seed + 3);
        for (auto& v : vs)
            v = std::abs(v) + 500.0;  // vs > 500

        random_fill(density, seed + 4);
        for (auto& d : density)
            d = std::abs(d) + 1000.0;  // density > 1000
    }
};

}  // anonymous namespace

TEST_CASE("CUDA element residual matches CPU — N=3 random", "[element][cuda]") {
    int N = 3;
    RandomElement elem(N, 42);

    std::vector<double> u(elem.n_node * 3, 0.0);
    random_fill(u, 99);

    std::vector<double> r_cpu(elem.n_node * 3, 0.0);
    std::vector<double> r_cuda(elem.n_node * 3, 0.0);

    // CPU reference
    compute_element_residual<BackendCPU>(1, elem.dxi_dx.data(), elem.jacobian.data(),
                                         elem.lambda_.data(), elem.mu_.data(), elem.D.data(),
                                         elem.w.data(), elem.ngll, u.data(), r_cpu.data());

    // CUDA result
    compute_element_residual<BackendCUDA>(1, elem.dxi_dx.data(), elem.jacobian.data(),
                                          elem.lambda_.data(), elem.mu_.data(), elem.D.data(),
                                          elem.w.data(), elem.ngll, u.data(), r_cuda.data());

    // Compare with relative tolerance
    // GPU uses atomicAdd which changes summation order vs CPU sequential loop.
    // This introduces machine-epsilon-level differences (~4e-16 relative).
    // Use relative tolerance to account for this fundamental parallel computation behavior.
    double max_rel_diff = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < r_cpu.size(); ++i) {
        double denom = std::max(std::abs(r_cpu[i]), 1.0e-14);
        double rel_diff = std::abs(r_cpu[i] - r_cuda[i]) / denom;
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
            max_idx = i;
        }
    }
    // Relative tolerance: 1e-12 allows machine-epsilon differences from
    // non-deterministic atomic summation order.
    REQUIRE(max_rel_diff < 1.0e-12);
}

TEST_CASE("CUDA element residual matches CPU — N=5 random", "[element][cuda]") {
    int N = 5;
    RandomElement elem(N, 123);

    std::vector<double> u(elem.n_node * 3, 0.0);
    random_fill(u, 456);

    std::vector<double> r_cpu(elem.n_node * 3, 0.0);
    std::vector<double> r_cuda(elem.n_node * 3, 0.0);

    compute_element_residual<BackendCPU>(1, elem.dxi_dx.data(), elem.jacobian.data(),
                                         elem.lambda_.data(), elem.mu_.data(), elem.D.data(),
                                         elem.w.data(), elem.ngll, u.data(), r_cpu.data());

    compute_element_residual<BackendCUDA>(1, elem.dxi_dx.data(), elem.jacobian.data(),
                                          elem.lambda_.data(), elem.mu_.data(), elem.D.data(),
                                          elem.w.data(), elem.ngll, u.data(), r_cuda.data());

    // Compare with relative tolerance
    double max_rel_diff = 0.0;
    for (size_t i = 0; i < r_cpu.size(); ++i) {
        double denom = std::max(std::abs(r_cpu[i]), 1.0e-14);
        double rel_diff = std::abs(r_cpu[i] - r_cuda[i]) / denom;
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    REQUIRE(max_rel_diff < 1.0e-12);
}

TEST_CASE("CUDA element residual — rigid body translation zero", "[element][cuda]") {
    int N = 3;
    RandomElement elem(N, 77);

    // Uniform translation
    std::vector<double> u(elem.n_node * 3, 0.0);
    for (int i = 0; i < elem.n_node; ++i) {
        u[i * 3 + 0] = 1.0;
        u[i * 3 + 1] = 2.0;
        u[i * 3 + 2] = 3.0;
    }

    std::vector<double> r(elem.n_node * 3, 0.0);
    compute_element_residual<BackendCUDA>(1, elem.dxi_dx.data(), elem.jacobian.data(),
                                          elem.lambda_.data(), elem.mu_.data(), elem.D.data(),
                                          elem.w.data(), elem.ngll, u.data(), r.data());

    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_THAT(r[i], WithinAbs(0.0, 1e-12));
    }
}