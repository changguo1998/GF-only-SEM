#pragma once

#include <cstddef>
#include <vector>

// CUDA host+device portability macro
#ifdef __CUDACC__
#define GF_HOST_DEVICE __host__ __device__
#else
#define GF_HOST_DEVICE
#endif

#include "gf/types.hpp"

// ============================================================================
// C-PML Strain Correction — named constants, structs, and enums
// All dimension-dependent values derived from NDIM = 3.
// See docs/_archive/specs/2026-07-21-cpml-strain-correction-design.md §2.
// ============================================================================
namespace CpmlStrain {

// --- Spatial dimension (root constant) ---
constexpr int NDIM = 3;

// --- Counts (derived from NDIM) ---
constexpr int NUM_DISPLACEMENT_COMPS = NDIM;
constexpr int NUM_DERIVATIVE_DIRS = NDIM;
constexpr int NUM_CONV_DIRECTIONS = NDIM;
constexpr int NUM_GRADIENT_COMPS = NDIM * NDIM;
constexpr int NUM_OFF_DIAG_GROUPS = NDIM;
constexpr int NUM_DIAG_GROUPS = NDIM;

// --- Coefficient strides ---
constexpr int LINGK_COEFS_PER_GROUP = 1 + NDIM;  // prefactor + NDIM memory dirs
constexpr int DIAG_COEFS_PER_GROUP = 2;          // NOT derived from NDIM
constexpr int COEFS_PER_NODE =
    NUM_OFF_DIAG_GROUPS * LINGK_COEFS_PER_GROUP + NUM_DIAG_GROUPS * DIAG_COEFS_PER_GROUP;

// --- Strain memory strides ---
constexpr int MEMORY_PER_GRADIENT = NDIM;
constexpr int MEMORY_PER_NODE = NUM_GRADIENT_COMPS * MEMORY_PER_GRADIENT;

// --- β convolution coefficient strides ---
constexpr int BETA_COEFS_PER_DIR = 3;  // NOT derived from NDIM
constexpr int BETA_COEFS_PER_NODE = NDIM * BETA_COEFS_PER_DIR;
constexpr int BETA_COEF0 = 0;
constexpr int BETA_COEF1 = 1;
constexpr int BETA_COEF2 = 2;

// --- Displacement memory strides ---
constexpr int DISPL_MEM_PER_NODE = NDIM * NDIM;

// --- Displacement field stride ---
constexpr int DISPL_FIELD_PER_NODE = NDIM;

// --- Offsets into pml_coef_strain (derived from group sizes) ---
constexpr int OFFSET_GRAD_WRT_X = 0;
constexpr int OFFSET_GRAD_WRT_Y = OFFSET_GRAD_WRT_X + LINGK_COEFS_PER_GROUP;
constexpr int OFFSET_GRAD_WRT_Z = OFFSET_GRAD_WRT_Y + LINGK_COEFS_PER_GROUP;
constexpr int OFFSET_DUX_DX = OFFSET_GRAD_WRT_Z + LINGK_COEFS_PER_GROUP;
constexpr int OFFSET_DUY_DY = OFFSET_DUX_DX + DIAG_COEFS_PER_GROUP;
constexpr int OFFSET_DUZ_DZ = OFFSET_DUY_DY + DIAG_COEFS_PER_GROUP;

// --- Structs ---
struct OffDiagonalCorrection {
    double gradient_prefactor;
    double memory_coef_conv_dir0;
    double memory_coef_conv_dir1;
    double memory_coef_conv_dir2;
};

struct DiagonalCorrection {
    double gradient_prefactor;
    double memory_coef_local_dir;
};

struct StrainCoefficients {
    OffDiagonalCorrection grad_wrt_x;
    OffDiagonalCorrection grad_wrt_y;
    OffDiagonalCorrection grad_wrt_z;
    DiagonalCorrection dux_dx;
    DiagonalCorrection duy_dy;
    DiagonalCorrection duz_dz;
};

// --- Enums ---
enum Gradient : int {
    DUX_DX = 0,
    DUX_DY = 1,
    DUX_DZ = 2,
    DUY_DX = 3,
    DUY_DY = 4,
    DUY_DZ = 5,
    DUZ_DX = 6,
    DUZ_DY = 7,
    DUZ_DZ = 8
};
enum Component : int { DUX = 0, DUY = 1, DUZ = 2 };
enum Direction : int { DX = 0, DY = 1, DZ = 2 };
enum ConvDir : int { CONV_X = 0, CONV_Y = 1, CONV_Z = 2 };

// --- Helpers ---
constexpr GF_HOST_DEVICE int gradient_of(int comp, int dir) {
    return comp * NUM_DERIVATIVE_DIRS + dir;
}

inline GF_HOST_DEVICE size_t strain_memory_offset(size_t node, int gradient, int conv_dir) {
    return node * MEMORY_PER_NODE + gradient * MEMORY_PER_GRADIENT + conv_dir;
}

inline GF_HOST_DEVICE StrainCoefficients load_strain_coefficients(const double* flat,
                                                                  size_t node) {
    const double* base = flat + node * COEFS_PER_NODE;
    StrainCoefficients c;
    c.grad_wrt_x = {base[OFFSET_GRAD_WRT_X + 0], base[OFFSET_GRAD_WRT_X + 1],
                    base[OFFSET_GRAD_WRT_X + 2], base[OFFSET_GRAD_WRT_X + 3]};
    c.grad_wrt_y = {base[OFFSET_GRAD_WRT_Y + 0], base[OFFSET_GRAD_WRT_Y + 1],
                    base[OFFSET_GRAD_WRT_Y + 2], base[OFFSET_GRAD_WRT_Y + 3]};
    c.grad_wrt_z = {base[OFFSET_GRAD_WRT_Z + 0], base[OFFSET_GRAD_WRT_Z + 1],
                    base[OFFSET_GRAD_WRT_Z + 2], base[OFFSET_GRAD_WRT_Z + 3]};
    c.dux_dx = {base[OFFSET_DUX_DX + 0], base[OFFSET_DUX_DX + 1]};
    c.duy_dy = {base[OFFSET_DUY_DY + 0], base[OFFSET_DUY_DY + 1]};
    c.duz_dz = {base[OFFSET_DUZ_DZ + 0], base[OFFSET_DUZ_DZ + 1]};
    return c;
}

}  // namespace CpmlStrain

namespace gf {

/// Apply legacy PML damping to the velocity field (backward compatibility).
///
/// For each DOF, the velocity is damped by a precomputed damping profile:
///   v[i] -= damping_profile[node] * v[i]
///
/// Used when C-PML data is not available (old partition files).
///
/// @param[in] damping_profile  per-GLL-node damping values [n_elem * NGLL^3]
/// @param[in] u                displacement field [n_dof]
/// @param[in,out] v            velocity field [n_dof], modified in place
/// @param[in] n_dof            total DOF count (= n_elem * NGLL^3 * 3)
void apply_pml_damping(const std::vector<double>& damping_profile, const std::vector<double>& u,
                       std::vector<double>& v, int n_dof);

// ---------------------------------------------------------------------------
// C-PML functions (recursive convolution, Wang et al. 2006)
// ---------------------------------------------------------------------------

/// Initialize C-PML memory state arrays.
///
/// Allocates and zero-initializes pml_displ_old, pml_displ_new,
/// rmemory_displ, and rmemory_strain based on the number of PML elements.
///
/// @param[in,out] part     RankData with C-PML coefficients loaded
/// @param[in] n_node       NGLL^3 (nodes per element)
void cpml_initialize(RankData& part, int n_node);

/// Save PML displacement field BEFORE Newmark predictor (uses old fields).
///
/// PML_displ_old = u + (1-2θ)/2 * dt * v + (1-θ)/2 * dt² * a
///
/// Matches SPECFEM3D update_displ_elastic_PML called BEFORE the predictor
/// (update_displacement_scheme.f90:296). Uses the OLD displacement, velocity,
/// and acceleration (pre-predictor state).
///
/// @param[in,out] part     RankData with PML displacement fields
/// @param[in] displacement Global displacement array [n_rank_dof] (old)
/// @param[in] velocity     Global velocity array [n_rank_dof] (old)
/// @param[in] acceleration Global acceleration array [n_rank_dof] (old)
/// @param[in] dt           Solver timestep
/// @param[in] n_node       NGLL^3
void cpml_save_displ_old(RankData& part, const std::vector<double>& displacement,
                         const std::vector<double>& velocity,
                         const std::vector<double>& acceleration, double dt, int n_node);

/// Save PML displacement field AFTER Newmark predictor (uses predicted fields).
///
/// PML_displ_new = u_tilde + (1-2θ)/2 * dt * v_pred
/// where v_pred = v + dt/2 * a (predicted velocity, matching SPECFEM3D's
/// in-place predictor veloc += dt/2*accel).
/// No acceleration term because accel = 0 after predictor.
///
/// Matches SPECFEM3D update_displ_elastic_PML called AFTER the predictor
/// (update_displacement_scheme.f90:305). Uses the PREDICTED displacement
/// (displacement_tilde) and PREDICTED velocity.
///
/// @param[in,out] part              RankData with PML displacement fields
/// @param[in] displacement_tilde    Predicted displacement [n_rank_dof]
/// @param[in] velocity              Global velocity array [n_rank_dof] (old)
/// @param[in] acceleration          Global acceleration array [n_rank_dof] (old)
/// @param[in] dt                    Solver timestep
/// @param[in] n_node                NGLL^3
void cpml_save_displ_new(RankData& part, const std::vector<double>& displacement_tilde,
                         const std::vector<double>& velocity,
                         const std::vector<double>& acceleration, double dt, int n_node);

/// Update C-PML displacement memory variables.
///
/// For each PML element and GLL node:
///   rmemory[d] = coef0_α[d] * rmemory[d]
///              + coef1_α[d] * PML_displ_new
///              + coef2_α[d] * PML_displ_old
///
/// @param[in,out] part     RankData with memory variables and coefficients
/// @param[in] n_node       NGLL^3
void cpml_update_displ_memory(RankData& part, int n_node);

/// Update C-PML strain memory variables via β convolution.
///
/// For each PML element and GLL node:
///   1. Compute 9 reference-space gradients of pml_displ_new and pml_displ_old
///      using the GLL derivative matrix D.
///   2. Transform to physical gradients using the inverse Jacobian dxi_dx.
///   3. Update rmemory_strain using β convolution coefficients.
///
/// @param[in,out] part     RankData with memory variables and coefficients
/// @param[in]     D        GLL derivative matrix [NGLL * NGLL]
/// @param[in]     weights  GLL quadrature weights [NGLL]
/// @param[in]     NGLL     Number of GLL points per axis
void cpml_update_strain_memory(RankData& part, const double* D, const double* weights, int NGLL);

/// Compute C-PML acceleration contribution and add to element-local residual.
///
/// For each PML element and GLL node:
///   residual += w * ρ * J * (Ā₁*v_pred + Ā₂*u_pred + Ā₃*mem_x + Ā₄*mem_y + Ā₅*mem_z)
///
/// Uses PREDICTED displacement (u_pred = displacement_tilde) and PREDICTED
/// velocity (v_pred = v + dt/2 * a), matching SPECFEM3D's in-place predictor
/// which modifies displ and veloc before pml_compute_accel_contribution.
///
/// @param[in] part               RankData with C-PML coefficients and memory
/// @param[in] displacement_tilde Predicted displacement [n_rank_dof]
/// @param[in] velocity           Global velocity array [n_rank_dof] (old)
/// @param[in] acceleration       Global acceleration array [n_rank_dof] (old)
/// @param[in] dt                 Solver timestep
/// @param[in] local_cell2rank_node  Element-to-rank-node mapping
/// @param[in] gll_weights        GLL quadrature weights [NGLL]
/// @param[in,out] residual       Element-local residual [n_local_cell * n_node * 3]
/// @param[in] n_local_cell       Number of local elements
/// @param[in] n_node             NGLL^3
void cpml_accel_contribution(const RankData& part, const std::vector<double>& displacement_tilde,
                             const std::vector<double>& velocity,
                             const std::vector<double>& acceleration, double dt,
                             const std::vector<int32_t>& local_cell2rank_node,
                             const std::vector<double>& gll_weights, std::vector<double>& residual,
                             int n_local_cell, int n_node);

}  // namespace gf
