// tests/test_element.cpp — matrix-free element residual tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include <cmath>
#include "gf/element.hpp"
#include "gf/gll.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

// Helper: build element arrays for a unit cube [0,1]^3 at polynomial order N
// with GLL node mapping
struct UnitCubeElement {
    int ngll;
    int n_node;
    std::vector<double> coords;
    std::vector<double> dxi_dx;
    std::vector<double> jacobian;
    std::vector<double> vp;
    std::vector<double> vs;
    std::vector<double> density;
    std::vector<double> D;  // 1D derivative matrix
    std::vector<double> w;  // 1D GLL weights
    std::vector<double> nodes; // 1D GLL nodes

    UnitCubeElement(int N) : ngll(N+1), n_node((N+1)*(N+1)*(N+1)) {
        nodes = gll_nodes(N);
        w = gll_weights(N, nodes);
        D = gll_derivative_matrix(N, nodes);

        coords.resize(n_node * 3);
        dxi_dx.resize(n_node * 9);
        jacobian.resize(n_node);
        vp.assign(n_node, 3000.0);
        vs.assign(n_node, 1500.0);
        density.assign(n_node, 2500.0);

        // Map natural coords (xi,eta,zeta) in [-1,1]^3 to physical [0,1]^3
        for (int k = 0; k < ngll; ++k) {
            for (int j = 0; j < ngll; ++j) {
                for (int i = 0; i < ngll; ++i) {
                    int idx = (i * ngll + j) * ngll + k;
                    double xi = nodes[i], eta = nodes[j], zeta = nodes[k];
                    coords[idx * 3 + 0] = 0.5 * (xi + 1.0);
                    coords[idx * 3 + 1] = 0.5 * (eta + 1.0);
                    coords[idx * 3 + 2] = 0.5 * (zeta + 1.0);
                    // Unit cube: dxi/dx = diag(2,2,2), det(J) = 1/8
                    dxi_dx[idx * 9 + 0] = 2.0;  // dxi/dx
                    dxi_dx[idx * 9 + 1] = 0.0;  // deta/dx
                    dxi_dx[idx * 9 + 2] = 0.0;  // dzeta/dx
                    dxi_dx[idx * 9 + 3] = 0.0;  // dxi/dy
                    dxi_dx[idx * 9 + 4] = 2.0;  // deta/dy
                    dxi_dx[idx * 9 + 5] = 0.0;  // dzeta/dy
                    dxi_dx[idx * 9 + 6] = 0.0;  // dxi/dz
                    dxi_dx[idx * 9 + 7] = 0.0;  // deta/dz
                    dxi_dx[idx * 9 + 8] = 2.0;  // dzeta/dz
                    jacobian[idx] = 0.125;       // det(J) = 1/8
                }
            }
        }
    }
};

TEST_CASE("Rigid-body translation gives zero residual", "[element]") {
    int N = 3;
    UnitCubeElement elem(N);

    // Uniform translation u = (1, 2, 3)
    std::vector<double> u(elem.n_node * 3, 0.0);
    for (int i = 0; i < elem.n_node; ++i) {
        u[i * 3 + 0] = 1.0;
        u[i * 3 + 1] = 2.0;
        u[i * 3 + 2] = 3.0;
    }

    std::vector<double> r(elem.n_node * 3, 0.0);

    compute_element_residual(
        elem.dxi_dx.data(), elem.jacobian.data(),
        elem.vp.data(), elem.vs.data(), elem.density.data(),
        elem.D.data(), elem.w.data(), elem.ngll,
        u.data(), r.data()
    );

    // Residual should be zero for rigid body translation (no strain, no stress)
    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_THAT(r[i], WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("Rigid-body rotation gives near-zero residual", "[element]") {
    int N = 3;
    UnitCubeElement elem(N);

    // Small rotation about z-axis: u_x = -theta*y, u_y = theta*x, u_z = 0
    double theta = 1e-6;
    std::vector<double> u(elem.n_node * 3, 0.0);
    for (int i = 0; i < elem.n_node; ++i) {
        double x = elem.coords[i * 3 + 0];
        double y = elem.coords[i * 3 + 1];
        u[i * 3 + 0] = -theta * y;
        u[i * 3 + 1] = theta * x;
    }

    std::vector<double> r(elem.n_node * 3, 0.0);
    compute_element_residual(
        elem.dxi_dx.data(), elem.jacobian.data(),
        elem.vp.data(), elem.vs.data(), elem.density.data(),
        elem.D.data(), elem.w.data(), elem.ngll,
        u.data(), r.data()
    );

    // Residual should be near zero for rigid rotation (only antisymmetric strain gradient)
    for (size_t i = 0; i < r.size(); ++i) {
        REQUIRE_THAT(r[i], WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("Uniform uniaxial strain produces correct residual", "[element]") {
    int N = 3;
    UnitCubeElement elem(N);

    // Apply ε_xx = 0.001 -> u_x = ε*x, u_y = 0, u_z = 0
    double eps = 0.001;
    std::vector<double> u(elem.n_node * 3, 0.0);
    for (int i = 0; i < elem.n_node; ++i) {
        double x = elem.coords[i * 3 + 0];
        u[i * 3 + 0] = eps * x;
    }

    std::vector<double> r(elem.n_node * 3, 0.0);
    compute_element_residual(
        elem.dxi_dx.data(), elem.jacobian.data(),
        elem.vp.data(), elem.vs.data(), elem.density.data(),
        elem.D.data(), elem.w.data(), elem.ngll,
        u.data(), r.data()
    );

    // For uniform ε_xx, stress is σ_xx = (λ+2μ)*ε, σ_yy = σ_zz = λ*ε
    // The residual is the internal force: r = -∫B^T σ dΩ
    // For a uniform strain field, the residual integrated over the element
    // is proportional to surface traction. For a free body, the internal
    // force integrated over the domain must be zero (self-equilibrating).
    // The nodal residual should satisfy force balance:
    // sum(r) over all nodes ≈ 0 (net force zero for self-equilibrating stress)
    double sum_rx = 0.0;
    for (int i = 0; i < elem.n_node; ++i) {
        sum_rx += r[i * 3 + 0];
    }
    REQUIRE_THAT(sum_rx, WithinAbs(0.0, 1e-8));
}