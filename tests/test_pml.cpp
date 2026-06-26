// tests/test_pml.cpp — PML damping tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "gf/pml.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

TEST_CASE("No damping in interior nodes", "[pml]") {
    int n_elem = 2;
    int ngll = 4;
    int n_node = n_elem * ngll * ngll * ngll;
    int n_dof = n_node * 3;

    // Zero damping profile everywhere (interior)
    std::vector<double> damping(n_node, 0.0);
    std::vector<double> u(n_dof, 0.5);
    std::vector<double> v(n_dof, 1.0);

    apply_pml_damping(damping, u, v, n_dof);

    // No damping applied - velocity unchanged
    for (int i = 0; i < n_dof; ++i) {
        REQUIRE_THAT(v[i], WithinAbs(1.0, 1e-15));
    }
}

TEST_CASE("Positive damping reduces velocity", "[pml]") {
    int n_elem = 1;
    int ngll = 4;
    int n_node = n_elem * ngll * ngll * ngll;
    int n_dof = n_node * 3;

    // Half the nodes have damping = 0.5, half have 0.0
    std::vector<double> damping(n_node, 0.0);
    std::vector<double> u(n_dof, 0.5);
    std::vector<double> v(n_dof, 2.0);
    for (int i = 0; i < n_node / 2; ++i) {
        damping[i] = 0.5;
    }

    apply_pml_damping(damping, u, v, n_dof);

    // Damped nodes: v_new = v_old - d * v_old = 2.0 - 0.5*2.0 = 1.0
    // Undamped nodes: v unchanged (2.0)
    int half_dof = (n_node / 2) * 3;
    for (int i = 0; i < half_dof; ++i) {
        REQUIRE_THAT(v[i], WithinAbs(1.0, 1e-15));
    }
    for (int i = half_dof; i < n_dof; ++i) {
        REQUIRE_THAT(v[i], WithinAbs(2.0, 1e-15));
    }
}

TEST_CASE("Full damping completely removes velocity", "[pml]") {
    int n_elem = 1;
    int ngll = 4;
    int n_node = n_elem * ngll * ngll * ngll;
    int n_dof = n_node * 3;

    // All nodes fully damped
    std::vector<double> damping(n_node, 1.0);
    std::vector<double> u(n_dof, 0.5);
    std::vector<double> v(n_dof, 3.0);

    apply_pml_damping(damping, u, v, n_dof);

    // v_new = v_old - 1.0 * v_old = 0.0
    for (int i = 0; i < n_dof; ++i) {
        REQUIRE_THAT(v[i], WithinAbs(0.0, 1e-15));
    }
}

TEST_CASE("Damping is per-node, not per-DOF", "[pml]") {
    // Verify that all 3 DOF at a node share the same damping coefficient
    int n_elem = 1;
    int ngll = 4;
    int n_node = n_elem * ngll * ngll * ngll;
    int n_dof = n_node * 3;

    std::vector<double> damping(n_node, 0.0);
    std::vector<double> u(n_dof, 0.0);
    std::vector<double> v_initial(n_dof, 1.0);

    // Damp first node only
    damping[0] = 0.8;

    std::vector<double> v = v_initial;
    apply_pml_damping(damping, u, v, n_dof);

    // First node (DOFs 0,1,2) should all be damped equally: v = 1-0.8 = 0.2
    REQUIRE_THAT(v[0], WithinAbs(0.2, 1e-15));
    REQUIRE_THAT(v[1], WithinAbs(0.2, 1e-15));
    REQUIRE_THAT(v[2], WithinAbs(0.2, 1e-15));
    // All others unchanged
    for (int i = 3; i < n_dof; ++i) {
        REQUIRE_THAT(v[i], WithinAbs(1.0, 1e-15));
    }
}