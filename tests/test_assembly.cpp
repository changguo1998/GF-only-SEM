// tests/test_assembly.cpp — assembly and RHS source scatter tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "gf/assembly.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

// Helper: build RankData with n_local_cell elements
static RankData make_rank(int n_local_cell, int ngll) {
    RankData rd;
    rd.n_local_cell = n_local_cell;
    rd.n_ghost_cell = 0;
    rd.n_total_cell = n_local_cell;
    rd.ngll = ngll;
    return rd;
}

TEST_CASE("assemble_residual copies element blocks to global", "[assembly]") {
    int ngll = 3;
    int n_local_cell = 2;
    int n_node = ngll * ngll * ngll;
    int n_dof_per_elem = n_node * 3;
    auto rd = make_rank(n_local_cell, ngll);

    // Build element residuals with distinct values per element
    std::vector<double> elem_r(n_local_cell * n_dof_per_elem, 0.0);
    for (int e = 0; e < n_local_cell; ++e) {
        for (int d = 0; d < n_dof_per_elem; ++d) {
            elem_r[e * n_dof_per_elem + d] = (e + 1) * 10.0 + d * 0.01;
        }
    }

    std::vector<double> global_r(n_local_cell * n_dof_per_elem, -1.0);
    assemble_residual(elem_r, rd, global_r);

    // Global residual should match element residual exactly (same layout)
    for (size_t i = 0; i < global_r.size(); ++i) {
        REQUIRE(global_r[i] == elem_r[i]);
    }
}

TEST_CASE("assemble_residual does not touch elements beyond n_local_cell", "[assembly]") {
    int ngll = 3;
    int n_local_cell = 2;
    int n_extra = 1;  // extra slot in global that should remain untouched
    int n_node = ngll * ngll * ngll;
    int n_dof_per_elem = n_node * 3;
    auto rd = make_rank(n_local_cell, ngll);

    std::vector<double> elem_r(n_local_cell * n_dof_per_elem, 1.0);
    std::vector<double> global_r((n_local_cell + n_extra) * n_dof_per_elem, -5.0);

    assemble_residual(elem_r, rd, global_r);

    // First n_local_cell elements should be overwritten
    for (int i = 0; i < n_local_cell * n_dof_per_elem; ++i) {
        REQUIRE(global_r[i] == 1.0);
    }
    // Extra slot should be untouched
    for (int i = n_local_cell * n_dof_per_elem; i < (n_local_cell + n_extra) * n_dof_per_elem;
         ++i) {
        REQUIRE(global_r[i] == -5.0);
    }
}

TEST_CASE("add_source_to_rhs adds force at correct DOF", "[assembly]") {
    int ngll = 4;
    int n_local_cell = 1;
    int n_node = ngll * ngll * ngll;
    int n_dof = n_local_cell * n_node * 3;
    auto rd = make_rank(n_local_cell, ngll);

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
    int n_local_cell = 1;
    int n_node = ngll * ngll * ngll;
    int n_dof = n_local_cell * n_node * 3;
    auto rd = make_rank(n_local_cell, ngll);

    std::vector<double> rhs(n_dof, 0.0);

    // Add force twice at the same location
    add_source_to_rhs(0, 0, 0, 0, 10.0, 0.0, 0.0, rd, rhs);
    add_source_to_rhs(0, 0, 0, 0, 20.0, 0.0, 0.0, rd, rhs);

    // DOF 0 should be accumulated: 10 + 20 = 30
    REQUIRE_THAT(rhs[0], WithinAbs(30.0, 1e-12));
}

TEST_CASE("add_source_to_rhs at different elements", "[assembly]") {
    int ngll = 4;
    int n_local_cell = 2;
    int n_node = ngll * ngll * ngll;
    int n_dof = n_local_cell * n_node * 3;
    auto rd = make_rank(n_local_cell, ngll);

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

// ===================================================================
// CG-SEM scatter / gather tests
// ===================================================================

TEST_CASE("scatter_to_rank accumulates shared-node contributions", "[scatter]") {
    // 2 elements, each with 2 GLL nodes (n_node=2).
    // local_cell2rank_node: elem0 n0→0, n1→1; elem1 n0→1, n1→2.
    // Global: [0] = elem0[0], [1] = elem0[1]+elem1[0], [2] = elem1[1].
    const int n_local_cell = 2;
    const int n_node = 2;

    std::vector<int32_t> local_cell2rank_node = {0, 1,   // elem 0: node0→glob0, node1→glob1
                                                 1, 2};  // elem 1: node0→glob1, node1→glob2
    int n_rank_node = 3;

    // elem0: n0=[10,20,30], n1=[40,50,60]
    // elem1: n0=[100,200,300], n1=[400,500,600]
    std::vector<double> local_cell_residual = {
        10.0,  20.0,  30.0,   // elem0 node0
        40.0,  50.0,  60.0,   // elem0 node1
        100.0, 200.0, 300.0,  // elem1 node0
        400.0, 500.0, 600.0   // elem1 node1
    };

    std::vector<double> rank_node_residual(n_rank_node * 3, -1.0);
    scatter_to_rank(local_cell_residual, local_cell2rank_node, n_local_cell, n_node,
                    rank_node_residual);

    // glob0: only elem0 node0 → [10, 20, 30]
    REQUIRE_THAT(rank_node_residual[0], WithinAbs(10.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[1], WithinAbs(20.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[2], WithinAbs(30.0, 1e-12));
    // glob1: elem0 node1 + elem1 node0 → [40+100=140, 50+200=250, 60+300=360]
    REQUIRE_THAT(rank_node_residual[3], WithinAbs(140.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[4], WithinAbs(250.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[5], WithinAbs(360.0, 1e-12));
    // glob2: only elem1 node1 → [400, 500, 600]
    REQUIRE_THAT(rank_node_residual[6], WithinAbs(400.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[7], WithinAbs(500.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[8], WithinAbs(600.0, 1e-12));
}

TEST_CASE("gather_from_rank copies global values to element-local", "[gather]") {
    const int n_local_cell = 2;
    const int n_node = 2;

    std::vector<int32_t> local_cell2rank_node = {0, 1,   // elem 0: node0→glob0, node1→glob1
                                                 1, 2};  // elem 1: node0→glob1, node1→glob2

    // Global displacement: glob0=[1,2,3], glob1=[4,5,6], glob2=[7,8,9]
    std::vector<double> global_disp = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<double> elem_disp(n_local_cell * n_node * 3, -1.0);

    gather_from_rank(global_disp, local_cell2rank_node, n_local_cell, n_node, elem_disp);

    // elem0 node0 → glob0 = [1,2,3]
    REQUIRE_THAT(elem_disp[0], WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(elem_disp[1], WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(elem_disp[2], WithinAbs(3.0, 1e-12));
    // elem0 node1 → glob1 = [4,5,6]
    REQUIRE_THAT(elem_disp[3], WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(elem_disp[4], WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(elem_disp[5], WithinAbs(6.0, 1e-12));
    // elem1 node0 → glob1 = [4,5,6]
    REQUIRE_THAT(elem_disp[6], WithinAbs(4.0, 1e-12));
    REQUIRE_THAT(elem_disp[7], WithinAbs(5.0, 1e-12));
    REQUIRE_THAT(elem_disp[8], WithinAbs(6.0, 1e-12));
    // elem1 node1 → glob2 = [7,8,9]
    REQUIRE_THAT(elem_disp[9], WithinAbs(7.0, 1e-12));
    REQUIRE_THAT(elem_disp[10], WithinAbs(8.0, 1e-12));
    REQUIRE_THAT(elem_disp[11], WithinAbs(9.0, 1e-12));
}

TEST_CASE("scatter-gather round-trip with shared nodes", "[scatter][gather]") {
    // 3 elements, 1 node each. local_cell2rank_node: elem0→0, elem1→1, elem2→0 (shared).
    const int n_local_cell = 3;
    const int n_node = 1;

    std::vector<int32_t> local_cell2rank_node = {0, 1, 0};
    int n_rank_node = 2;

    // Global: glob0=[10,20,30], glob1=[40,50,60]
    std::vector<double> global_disp = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
    std::vector<double> elem_disp(n_local_cell * n_node * 3, 0.0);

    // Gather
    gather_from_rank(global_disp, local_cell2rank_node, n_local_cell, n_node, elem_disp);

    // elem0 and elem2 both have glob0 values
    REQUIRE_THAT(elem_disp[0], WithinAbs(10.0, 1e-12));  // elem0 x
    REQUIRE_THAT(elem_disp[6], WithinAbs(10.0, 1e-12));  // elem2 x

    // Set element-local to known values (like residual computation)
    for (int d = 0; d < 3; ++d) {
        elem_disp[0 + d] = 100.0 + d;  // elem0 = [100, 101, 102]
        elem_disp[3 + d] = 200.0 + d;  // elem1 = [200, 201, 202]
        elem_disp[6 + d] = 300.0 + d;  // elem2 = [300, 301, 302]
    }

    // Scatter back: glob0 = elem0 + elem2, glob1 = elem1
    std::vector<double> rank_node_residual(n_rank_node * 3, 0.0);
    scatter_to_rank(elem_disp, local_cell2rank_node, n_local_cell, n_node, rank_node_residual);

    // glob0: [100+300=400, 101+301=402, 102+302=404]
    REQUIRE_THAT(rank_node_residual[0], WithinAbs(400.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[1], WithinAbs(402.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[2], WithinAbs(404.0, 1e-12));
    // glob1: [200, 201, 202]
    REQUIRE_THAT(rank_node_residual[3], WithinAbs(200.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[4], WithinAbs(201.0, 1e-12));
    REQUIRE_THAT(rank_node_residual[5], WithinAbs(202.0, 1e-12));
}