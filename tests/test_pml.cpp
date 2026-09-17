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

// ============================================================================
// C-PML strain correction tests (CpmlStrain namespace + cpml_update_strain_memory)
// ============================================================================

#include "gf/types.hpp"

using namespace CpmlStrain;

TEST_CASE("CpmlStrain constants are dimension-consistent", "[pml][cpml]") {
    // All counts derive from NDIM = 3
    REQUIRE(NUM_DISPLACEMENT_COMPS == 3);
    REQUIRE(NUM_DERIVATIVE_DIRS == 3);
    REQUIRE(NUM_CONV_DIRECTIONS == 3);
    REQUIRE(NUM_GRADIENT_COMPS == 9);
    REQUIRE(NUM_OFF_DIAG_GROUPS == 3);
    REQUIRE(NUM_DIAG_GROUPS == 3);

    // Coefficient strides
    REQUIRE(LINGK_COEFS_PER_GROUP == 4);       // 1 prefactor + 3 memory dirs
    REQUIRE(DIAG_COEFS_PER_GROUP == 2);        // 1 prefactor + 1 memory dir
    REQUIRE(COEFS_PER_NODE == 3 * 4 + 3 * 2);  // 12 + 6 = 18

    // Memory strides
    REQUIRE(MEMORY_PER_GRADIENT == 3);
    REQUIRE(LIJK_MEMORY_PER_NODE == 9 * 3);  // 27
    REQUIRE(LX_LY_LZ_MEMORY_PER_NODE == 12);
    REQUIRE(MEMORY_PER_NODE == 27 + 12);  // 39

    // β coefficient strides
    REQUIRE(BETA_COEFS_PER_DIR == 3);
    REQUIRE(BETA_COEFS_PER_NODE == 9);
    REQUIRE(BETA_COEF0 == 0);
    REQUIRE(BETA_COEF1 == 1);
    REQUIRE(BETA_COEF2 == 2);

    // Offsets are strictly increasing
    REQUIRE(OFFSET_GRAD_WRT_X == 0);
    REQUIRE(OFFSET_GRAD_WRT_Y == 4);
    REQUIRE(OFFSET_GRAD_WRT_Z == 8);
    REQUIRE(OFFSET_DUX_DX == 12);
    REQUIRE(OFFSET_DUY_DY == 14);
    REQUIRE(OFFSET_DUZ_DZ == 16);

    // Enum values
    REQUIRE(static_cast<int>(DUX_DX) == 0);
    REQUIRE(static_cast<int>(DUZ_DZ) == 8);
    REQUIRE(static_cast<int>(DUX) == 0);
    REQUIRE(static_cast<int>(DUZ) == 2);
    REQUIRE(static_cast<int>(DX) == 0);
    REQUIRE(static_cast<int>(DZ) == 2);
}

TEST_CASE("gradient_of maps component+dir to correct gradient index", "[pml][cpml]") {
    // gradient_of(component, direction) = component * 3 + direction
    REQUIRE(gradient_of(DUX, DX) == DUX_DX);  // 0*3+0 = 0
    REQUIRE(gradient_of(DUX, DY) == DUX_DY);  // 0*3+1 = 1
    REQUIRE(gradient_of(DUX, DZ) == DUX_DZ);  // 0*3+2 = 2
    REQUIRE(gradient_of(DUY, DX) == DUY_DX);  // 1*3+0 = 3
    REQUIRE(gradient_of(DUY, DY) == DUY_DY);  // 1*3+1 = 4
    REQUIRE(gradient_of(DUY, DZ) == DUY_DZ);  // 1*3+2 = 5
    REQUIRE(gradient_of(DUZ, DX) == DUZ_DX);  // 2*3+0 = 6
    REQUIRE(gradient_of(DUZ, DY) == DUZ_DY);  // 2*3+1 = 7
    REQUIRE(gradient_of(DUZ, DZ) == DUZ_DZ);  // 2*3+2 = 8
}

TEST_CASE("strain_memory_offset produces unique offset per node", "[pml][cpml]") {
    // Base: node * MEMORY_PER_NODE + gradient * MEMORY_PER_GRADIENT + conv_dir
    size_t offset_0_0_0 = strain_memory_offset(0, DUX_DX, CONV_X);
    REQUIRE(offset_0_0_0 == 0);

    size_t offset_0_0_1 = strain_memory_offset(0, DUX_DX, CONV_Y);
    REQUIRE(offset_0_0_1 == 1);

    size_t offset_0_1_0 = strain_memory_offset(0, DUX_DY, CONV_X);
    REQUIRE(offset_0_1_0 == 3);  // grad 1 * 3 = 3

    size_t offset_0_8_2 = strain_memory_offset(0, DUZ_DZ, CONV_Z);
    REQUIRE(offset_0_8_2 == 8 * 3 + 2);  // 24 + 2 = 26

    // Next node starts after the 27 lijk and 12 lx/ly/lz memory values.
    size_t offset_1_0_0 = strain_memory_offset(1, DUX_DX, CONV_X);
    REQUIRE(offset_1_0_0 == 39);
}

TEST_CASE("load_strain_coefficients loads all 6 correction groups", "[pml][cpml]") {
    std::vector<double> flat(COEFS_PER_NODE);
    flat[OFFSET_GRAD_WRT_X + 0] = 1.1;  // prefactor
    flat[OFFSET_GRAD_WRT_X + 1] = 1.2;  // mem dir0
    flat[OFFSET_GRAD_WRT_X + 2] = 1.3;  // mem dir1
    flat[OFFSET_GRAD_WRT_X + 3] = 1.4;  // mem dir2

    flat[OFFSET_DUX_DX + 0] = 4.1;  // prefactor
    flat[OFFSET_DUX_DX + 1] = 4.2;  // mem local dir

    flat[OFFSET_DUZ_DZ + 0] = 6.1;
    flat[OFFSET_DUZ_DZ + 1] = 6.2;

    StrainCoefficients c = load_strain_coefficients(flat.data(), 0);

    REQUIRE(c.grad_wrt_x.gradient_prefactor == 1.1);
    REQUIRE(c.grad_wrt_x.memory_coef_conv_dir0 == 1.2);
    REQUIRE(c.grad_wrt_x.memory_coef_conv_dir1 == 1.3);
    REQUIRE(c.grad_wrt_x.memory_coef_conv_dir2 == 1.4);

    REQUIRE(c.dux_dx.gradient_prefactor == 4.1);
    REQUIRE(c.dux_dx.memory_coef_local_dir == 4.2);

    REQUIRE(c.duz_dz.gradient_prefactor == 6.1);
    REQUIRE(c.duz_dz.memory_coef_local_dir == 6.2);
}

TEST_CASE("cpml_update_strain_memory updates PML element memory", "[pml][cpml]") {
    RankData part;
    int ngll = 2;
    int n_node = ngll * ngll * ngll;  // 8
    part.n_local_cell = 1;
    part.ngll = ngll;
    part.has_cpml = true;

    part.pml_region = {1};  // PML element

    // Allocate β coefficients: 9 per node, set to produce non-zero update
    // β0=0.5, β1=0.3, β2=0.2 for each direction
    part.pml_coef_beta.assign(n_node * 9, 0.0);
    for (int n = 0; n < n_node; ++n) {
        for (int d = 0; d < 3; ++d) {
            part.pml_coef_beta[n * 9 + d * 3 + 0] = 0.5;
            part.pml_coef_beta[n * 9 + d * 3 + 1] = 0.3;
            part.pml_coef_beta[n * 9 + d * 3 + 2] = 0.2;
        }
    }
    // LX/LY/LZ memory uses the alpha convolution coefficients.
    part.pml_coef_alpha = part.pml_coef_beta;

    // Identity Jacobian (physical gradient = reference gradient)
    part.dxi_dx.assign(n_node * 9, 0.0);
    for (int n = 0; n < n_node; ++n) {
        part.dxi_dx[n * 9 + 0] = 1.0;
        part.dxi_dx[n * 9 + 4] = 1.0;
        part.dxi_dx[n * 9 + 8] = 1.0;
    }

    // Non-zero PML displacement fields
    part.pml_displ_old.assign(n_node * 3, 0.1);
    part.pml_displ_new.assign(n_node * 3, 0.2);
    part.rmemory_strain.assign(n_node * MEMORY_PER_NODE, 0.0);

    // Simple non-zero GLL derivative matrix for NGLL=2
    // D = [[0.5, 0.5], [0.5, 0.5]] — constant derivative approximation
    std::vector<double> D(ngll * ngll, 0.5);
    std::vector<double> weights(ngll, 1.0);

    cpml_update_strain_memory(part, D.data(), weights.data(), ngll);

    // Verify: strain memory is no longer all zero
    bool has_nonzero = false;
    for (double v : part.rmemory_strain) {
        if (std::abs(v) > 1e-15) {
            has_nonzero = true;
            break;
        }
    }
    REQUIRE(has_nonzero);
}

TEST_CASE("cpml_update_strain_memory skips interior elements", "[pml][cpml]") {
    RankData part;
    int ngll = 2;
    int n_node = ngll * ngll * ngll;
    part.n_local_cell = 1;
    part.ngll = ngll;
    part.has_cpml = true;

    part.pml_region = {0};  // interior — NOT PML

    part.pml_coef_beta.assign(n_node * 9, 0.5);
    part.dxi_dx.assign(n_node * 9, 0.0);
    for (int n = 0; n < n_node; ++n) {
        part.dxi_dx[n * 9 + 0] = 1.0;
        part.dxi_dx[n * 9 + 4] = 1.0;
        part.dxi_dx[n * 9 + 8] = 1.0;
    }
    part.pml_displ_old.assign(n_node * 3, 0.1);
    part.pml_displ_new.assign(n_node * 3, 0.2);
    part.rmemory_strain.assign(n_node * MEMORY_PER_NODE, 0.0);

    std::vector<double> D(ngll * ngll, 0.5);
    std::vector<double> weights(ngll, 1.0);

    cpml_update_strain_memory(part, D.data(), weights.data(), ngll);

    // Interior element: strain memory should remain zero
    for (double v : part.rmemory_strain) {
        REQUIRE(std::abs(v) < 1e-15);
    }
}

TEST_CASE("cpml_update_strain_memory no-op when has_cpml is false", "[pml][cpml]") {
    RankData part;
    int ngll = 2;
    int n_node = ngll * ngll * ngll;
    part.n_local_cell = 1;
    part.ngll = ngll;
    part.has_cpml = false;  // C-PML not enabled

    part.pml_region = {1};
    part.pml_coef_beta.assign(n_node * 9, 0.5);
    part.pml_displ_old.assign(n_node * 3, 0.1);
    part.pml_displ_new.assign(n_node * 3, 0.2);
    part.rmemory_strain.assign(n_node * MEMORY_PER_NODE, 0.0);

    std::vector<double> D(ngll * ngll, 0.5);
    std::vector<double> weights(ngll, 1.0);

    cpml_update_strain_memory(part, D.data(), weights.data(), ngll);

    // Memory untouched
    for (double v : part.rmemory_strain) {
        REQUIRE(std::abs(v) < 1e-15);
    }
}
