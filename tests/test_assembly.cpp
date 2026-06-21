// tests/test_assembly.cpp — assembly and RHS source scatter tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>
#include "gf/assembly.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

// Helper: build RankData with n_local elements
static RankData make_rank(int n_local, int ngll) {
    RankData rd;
    rd.n_local_elem = n_local;
    rd.n_ghost_elem = 0;
    rd.n_total_elem = n_local;
    rd.ngll = ngll;
    return rd;
}

TEST_CASE("assemble_residual copies element blocks to global", "[assembly]") {
    int ngll = 3;
    int n_local = 2;
    int n_node = ngll * ngll * ngll;
    int n_dof_per_elem = n_node * 3;
    auto rd = make_rank(n_local, ngll);

    // Build element residuals with distinct values per element
    std::vector<double> elem_r(n_local * n_dof_per_elem, 0.0);
    for (int e = 0; e < n_local; ++e) {
        for (int d = 0; d < n_dof_per_elem; ++d) {
            elem_r[e * n_dof_per_elem + d] = (e + 1) * 10.0 + d * 0.01;
        }
    }

    std::vector<double> global_r(n_local * n_dof_per_elem, -1.0);
    assemble_residual(elem_r, rd, global_r);

    // Global residual should match element residual exactly (same layout)
    for (size_t i = 0; i < global_r.size(); ++i) {
        REQUIRE(global_r[i] == elem_r[i]);
    }
}

TEST_CASE("assemble_residual does not touch elements beyond n_local", "[assembly]") {
    int ngll = 3;
    int n_local = 2;
    int n_extra = 1; // extra slot in global that should remain untouched
    int n_node = ngll * ngll * ngll;
    int n_dof_per_elem = n_node * 3;
    auto rd = make_rank(n_local, ngll);

    std::vector<double> elem_r(n_local * n_dof_per_elem, 1.0);
    std::vector<double> global_r((n_local + n_extra) * n_dof_per_elem, -5.0);

    assemble_residual(elem_r, rd, global_r);

    // First n_local elements should be overwritten
    for (int i = 0; i < n_local * n_dof_per_elem; ++i) {
        REQUIRE(global_r[i] == 1.0);
    }
    // Extra slot should be untouched
    for (int i = n_local * n_dof_per_elem; i < (n_local + n_extra) * n_dof_per_elem; ++i) {
        REQUIRE(global_r[i] == -5.0);
    }
}

TEST_CASE("add_source_to_rhs adds force at correct DOF", "[assembly]") {
    int ngll = 4;
    int n_local = 1;
    int n_node = ngll * ngll * ngll;
    int n_dof = n_local * n_node * 3;
    auto rd = make_rank(n_local, ngll);

    std::vector<double> rhs(n_dof, 0.0);

    // Add force at element 0, GLL (1,2,3), force = (100, 200, 300)
    add_source_to_rhs(0, 1, 2, 3, 100.0, 200.0, 300.0, rd, rhs);

    // Compute expected DOF index
    int node_idx = (1 * ngll + 2) * ngll + 3;
    int dof_base = node_idx * 3;

    REQUIRE_THAT(rhs[dof_base + 0], WithinAbs(100.0, 1e-12));
    REQUIRE_THAT(rhs[dof_base + 1], WithinAbs(200.0, 1e-12));
    REQUIRE_THAT(rhs[dof_base + 2], WithinAbs(300.0, 1e-12));

    // All other DOFs should be zero
    double sum_other = 0.0;
    for (int i = 0; i < n_dof; ++i) {
        if (i < dof_base || i >= dof_base + 3) {
            sum_other += std::abs(rhs[i]);
        }
    }
    REQUIRE_THAT(sum_other, WithinAbs(0.0, 1e-12));
}

TEST_CASE("add_source_to_rhs accumulates across calls", "[assembly]") {
    int ngll = 4;
    int n_local = 1;
    int n_node = ngll * ngll * ngll;
    int n_dof = n_local * n_node * 3;
    auto rd = make_rank(n_local, ngll);

    std::vector<double> rhs(n_dof, 0.0);

    // Add force twice at the same location
    add_source_to_rhs(0, 0, 0, 0, 10.0, 0.0, 0.0, rd, rhs);
    add_source_to_rhs(0, 0, 0, 0, 20.0, 0.0, 0.0, rd, rhs);

    // DOF 0 should be accumulated: 10 + 20 = 30
    REQUIRE_THAT(rhs[0], WithinAbs(30.0, 1e-12));
}

TEST_CASE("add_source_to_rhs at different elements", "[assembly]") {
    int ngll = 4;
    int n_local = 2;
    int n_node = ngll * ngll * ngll;
    int n_dof = n_local * n_node * 3;
    auto rd = make_rank(n_local, ngll);

    std::vector<double> rhs(n_dof, 0.0);

    // Add to element 0, node 0: fx = 50
    add_source_to_rhs(0, 0, 0, 0, 50.0, 0.0, 0.0, rd, rhs);

    // Add to element 1, node 0: fy = 75
    add_source_to_rhs(1, 0, 0, 0, 0.0, 75.0, 0.0, rd, rhs);

    // Element 0, first DOF = 50 (x-component)
    REQUIRE_THAT(rhs[0], WithinAbs(50.0, 1e-12));

    // Element 1, second DOF = 75 (y-component)
    int elem1_base = n_node * 3;
    REQUIRE_THAT(rhs[elem1_base + 1], WithinAbs(75.0, 1e-12));
}