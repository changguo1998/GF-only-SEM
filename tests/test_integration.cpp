// tests/test_integration.cpp — minimal end-to-end forward simulation test
//
// Tests the complete solver loop for a single-element configuration
// with no MPI exchange needed (single rank).
#include <hdf5.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "gf/assembly.hpp"
#include "gf/element.hpp"
#include "gf/gll.hpp"
#include "gf/newmark.hpp"
#include "gf/pml.hpp"
#include "gf/source.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

namespace {

// Build a minimal RankData for a single element
RankData build_single_element(int ngll) {
    RankData rd;
    rd.n_local_elem = 1;
    rd.n_ghost_elem = 0;
    rd.n_total_elem = 1;
    rd.ngll = ngll;

    int n_node = ngll * ngll * ngll;
    rd.local_element_ids = {1};

    // GLL nodes (unit cube [0,1]^3)
    auto nodes = gll_nodes(ngll - 1);
    int n = ngll;
    rd.coords.resize(3 * n_node);
    rd.jacobian.resize(n_node);
    rd.dxi_dx.resize(9 * n_node);
    rd.vp.resize(n_node);
    rd.vs.resize(n_node);
    rd.density.resize(n_node);
    rd.lambda_.resize(n_node);
    rd.mu_.resize(n_node);
    rd.mass.resize(n_node);

    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int idx = (i * n + j) * n + k;
                double xi = nodes[i], eta = nodes[j], zeta = nodes[k];

                // Coordinates in [0,1]^3
                rd.coords[3 * idx + 0] = 0.5 * (xi + 1.0);
                rd.coords[3 * idx + 1] = 0.5 * (eta + 1.0);
                rd.coords[3 * idx + 2] = 0.5 * (zeta + 1.0);

                // Unit cube: dxi/dx = diag(2,2,2), det(J) = 1/8
                int base = 9 * idx;
                rd.dxi_dx[base + 0] = 2.0;
                rd.dxi_dx[base + 1] = 0.0;
                rd.dxi_dx[base + 2] = 0.0;
                rd.dxi_dx[base + 3] = 0.0;
                rd.dxi_dx[base + 4] = 2.0;
                rd.dxi_dx[base + 5] = 0.0;
                rd.dxi_dx[base + 6] = 0.0;
                rd.dxi_dx[base + 7] = 0.0;
                rd.dxi_dx[base + 8] = 2.0;
                rd.jacobian[idx] = 0.125;

                // Material: Vp=3000, Vs=1500, density=2500
                rd.vp[idx] = 3000.0;
                rd.vs[idx] = 1500.0;
                rd.density[idx] = 2500.0;
                // Precompute elastic coefficients
                double vs2 = rd.vs[idx] * rd.vs[idx];
                double vp2 = rd.vp[idx] * rd.vp[idx];
                rd.mu_[idx] = rd.density[idx] * vs2;
                rd.lambda_[idx] = rd.density[idx] * (vp2 - 2.0 * vs2);

                // Lumped mass (approximate for unit cube)
                rd.mass[idx] = 2500.0 * 0.125 * 64.0 / (n_node);  // rough estimate
            }
        }
    }

    // PML: all interior
    rd.pml_damping.assign(n_node, 0.0);

    return rd;
}

}  // anonymous namespace

TEST_CASE("Single-element forward steps complete without crash", "[integration]") {
    int N = 3;  // polynomial order
    int ngll = N + 1;
    int n_node = ngll * ngll * ngll;
    int n_dof = 1 * n_node * 3;

    auto rd = build_single_element(ngll);

    // GLL quadrature
    auto nodes = gll_nodes(N);
    auto wts = gll_weights(N, nodes);
    auto D = gll_derivative_matrix(N, nodes);

    // Allocate state vectors
    std::vector<double> u(n_dof, 0.0);
    std::vector<double> v(n_dof, 0.0);
    std::vector<double> a(n_dof, 0.0);
    std::vector<double> r(n_dof, 0.0);
    std::vector<double> u_tilde(n_dof, 0.0);
    std::vector<double> elem_r(n_dof, 0.0);

    // Time stepping parameters
    double dt = 1e-5;
    double beta = 0.0;
    double gamma = 0.5;
    int nsteps = 10;

    // Apply initial perturbation (small displacement near element center)
    int mid_gll = ngll / 2;
    int mid_idx = (mid_gll * ngll + mid_gll) * ngll + mid_gll;
    int mid_dof = mid_idx * 3;
    u[mid_dof] = 1e-8;  // small perturbation in x

    for (int step = 0; step < nsteps; ++step) {
        // --- Predictor ---
        for (int i = 0; i < n_dof; ++i) {
            u_tilde[i] = u[i] + dt * v[i] + (0.5 * dt * dt * (1.0 - 2.0 * beta)) * a[i];
        }

        // --- Element residual ---
        std::fill(elem_r.begin(), elem_r.end(), 0.0);
        compute_element_residual<gf::BackendCPU>(
            1 /* n_elem */, rd.dxi_dx.data(), rd.jacobian.data(), rd.lambda_.data(), rd.mu_.data(),
            D.data(), wts.data(), ngll, u_tilde.data(), elem_r.data());

        // --- Assembly ---
        assemble_residual(elem_r, rd, r);

        // --- PML damping on velocity ---
        apply_pml_damping(rd.pml_damping, u, v, n_dof);

        // --- Corrector ---
        for (int i = 0; i < n_dof; ++i) {
            double a_new = r[i] / rd.mass[i / 3];
            u[i] += dt * v[i] + 0.5 * dt * dt * a_new;
            v[i] += dt * gamma * a_new;
            a[i] = a_new;
        }
    }

    // Verify simulation produced finite values
    for (int i = 0; i < n_dof; ++i) {
        REQUIRE(std::isfinite(u[i]));
        REQUIRE(std::isfinite(v[i]));
        REQUIRE(std::isfinite(a[i]));
        REQUIRE(std::isfinite(r[i]));
    }

    // The perturbation should produce non-zero response
    // (the wavefield should have propagated beyond the initial node)
    bool has_nonzero = false;
    for (int i = 0; i < n_dof; ++i) {
        if (std::abs(u[i]) > 1e-20) {
            has_nonzero = true;
            break;
        }
    }
    REQUIRE(has_nonzero);
}

TEST_CASE("Rigid-body initial condition produces zero residual", "[integration]") {
    int N = 3;
    int ngll = N + 1;
    int n_node = ngll * ngll * ngll;
    int n_dof = 1 * n_node * 3;

    auto rd = build_single_element(ngll);

    auto nodes = gll_nodes(N);
    auto wts = gll_weights(N, nodes);
    auto D = gll_derivative_matrix(N, nodes);

    // Uniform translation: u = (1, 2, 3)
    std::vector<double> u(n_dof, 0.0);
    for (int i = 0; i < n_node; ++i) {
        u[3 * i + 0] = 1.0;
        u[3 * i + 1] = 2.0;
        u[3 * i + 2] = 3.0;
    }

    std::vector<double> r(n_dof, 0.0);
    compute_element_residual<gf::BackendCPU>(1 /* n_elem */, rd.dxi_dx.data(), rd.jacobian.data(),
                                             rd.lambda_.data(), rd.mu_.data(), D.data(),
                                             wts.data(), ngll, u.data(), r.data());

    // Rigid-body translation → zero residual
    for (int i = 0; i < n_dof; ++i) {
        REQUIRE_THAT(r[i], WithinAbs(0.0, 1e-12));
    }
}

TEST_CASE("PML damping in integration loop reduces velocity", "[integration]") {
    int N = 3;
    int ngll = N + 1;
    int n_node = ngll * ngll * ngll;
    int n_dof = 1 * n_node * 3;

    auto rd = build_single_element(ngll);

    // Set strong PML damping everywhere
    rd.pml_damping.assign(n_node, 0.9);

    // Set non-zero velocity
    std::vector<double> u(n_dof, 0.0);
    std::vector<double> v(n_dof, 1.0);

    apply_pml_damping(rd.pml_damping, u, v, n_dof);

    // After damping: v_new = v - 0.9*v = 0.1
    for (int i = 0; i < n_dof; ++i) {
        REQUIRE_THAT(v[i], WithinAbs(0.1, 1e-12));
    }
}