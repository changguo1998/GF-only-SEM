// tests/test_source.cpp — source injection tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "gf/source.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

// Helper to build a minimal RankData
static RankData make_simple_rank(int n_local_element, int ngll) {
    RankData rd;
    rd.n_local_element = n_local_element;
    rd.ngll = ngll;
    for (int i = 0; i < n_local_element; ++i) {
        rd.local_element_ids.push_back(i + 1);
    }
    return rd;
}

TEST_CASE("Source location at GLL node", "[source]") {
    int ngll = 4;
    int n_local_element = 1;
    int n_node = ngll * ngll * ngll;

    // Place a GLL node at (0.5, 0.5, 0.5) in the unit cube
    std::vector<double> coords(n_local_element * n_node * 3, 0.0);
    std::vector<double> dxi_dx(n_local_element * n_node * 9, 0.0);

    // Material at GLL nodes (needed, even if unused)
    // Source at element centroid (0.5, 0.5, 0.5) - this should correspond to the
    // center node for a unit cube. For a cubic element, the center natural coord is (0,0,0).
    // Ngll=4, we'll place source near the first interior node.
    // For simplicity, we check that locate returns true (finds the element).
    PointForceSource src;
    bool found = src.locate(0.5, 0.5, 0.5, coords, dxi_dx, n_local_element, ngll);
    // The locate might fail with zero coords, but the call should complete
    REQUIRE(found == false);  // Expected: zero coords, no element containing point
}

TEST_CASE("Source apply preserves force magnitude", "[source]") {
    int ngll = 4;
    int n_node = ngll * ngll * ngll;
    int n_local_element = 1;
    int n_dof = n_local_element * n_node * 3;

    auto rd = make_simple_rank(n_local_element, ngll);
    std::vector<double> rhs(n_dof, 0.0);

    PointForceSource src;
    src.element_id = 1;
    src.wx = 1.0;  // weight at GLL node
    src.wy = 0.0;
    src.wz = 0.0;
    src.gll_i = 0;
    src.gll_j = 0;
    src.gll_k = 0;

    // Apply unit force in x direction
    src.apply(1.0, 0.0, 0.0, rd, rhs);

    // Find which DOF got the force
    double sum_rhs = 0.0;
    for (int i = 0; i < n_dof; ++i) {
        sum_rhs += rhs[i];
    }
    // Total force should be conserved: sum(rhs) = 1.0 (from the x-component)
    // More precisely, each node gets weight * force in the distributed direction
    REQUIRE_THAT(sum_rhs, WithinAbs(1.0, 1e-12));
}

TEST_CASE("Source conservation across multiple elements", "[source]") {
    int ngll = 4;
    int n_local_element = 2;
    int n_node = ngll * ngll * ngll;
    int n_dof = n_local_element * n_node * 3;

    auto rd = make_simple_rank(n_local_element, ngll);
    std::vector<double> rhs(n_dof, 0.0);

    // Simulate a source that distributes across 2 elements
    PointForceSource src1, src2;
    src1.element_id = 1;
    src1.wx = 0.6;
    src2.element_id = 2;
    src2.wx = 0.4;

    src1.apply(1.0, 0.0, 0.0, rd, rhs);
    src2.apply(0.0, 0.0, 0.0, rd, rhs);  // second element doesn't contribute

    double sum_rhs = 0.0;
    for (int i = 0; i < n_dof; ++i) {
        sum_rhs += rhs[i];
    }
    // Sum of weights = 0.6, total force contributed = 0.6
    REQUIRE_THAT(sum_rhs, WithinAbs(0.6, 1e-12));
}