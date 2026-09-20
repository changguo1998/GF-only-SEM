/**
 * @file attenuation.hpp
 * @brief SLS (Standard Linear Solid) viscoelastic attenuation constants and helpers.
 *
 * Spec: docs/_archive/specs/2026-07-21-sls-viscoelastic-design.md §4
 */

#ifndef GF_ATTENUATION_HPP_
#define GF_ATTENUATION_HPP_

#include <cmath>
#include <cstddef>

// ---------------------------------------------------------------------------
// SLS (Standard Linear Solid) Viscoelastic Attenuation
// ---------------------------------------------------------------------------
// References:
//   - Komatitsch & Tromp (1999), "Introduction to the spectral element method..."
//   - SPECFEM3D attenuation implementation (compute_forces_viscoelastic.F90)

namespace SLS {

// --- Compile-time constants ---
constexpr int N_SLS = 3;             // number of relaxation mechanisms per GLL node
constexpr int NDIM = 3;              // spatial dimensions
constexpr int VOIGT_COMPONENTS = 6;  // independent symmetric stress/strain components
constexpr int MEMORY_PER_NODE = N_SLS * VOIGT_COMPONENTS;  // 18 doubles per node
constexpr int TAU_PER_NODE = N_SLS * 3;                    // τ_σ + τ_ε,mu + τ_ε,kappa
constexpr int FORCING_WEIGHTS = 2;  // previous/current strain weights per mechanism

enum StrainComponent : int {
    DEVIATORIC_XX = 0,
    DEVIATORIC_YY = 1,
    TRACE = 2,
    XY = 3,
    XZ = 4,
    YZ = 5,
};

// --- Voigt index ---
// Maps 3×3 symmetric tensor indices (l,m) to 1D Voigt index:
//   (0,0)=0  (1,1)=1  (2,2)=2  (0,1)=3  (0,2)=4  (1,2)=5
// Valid only for l,m in {0,1,2}.
inline constexpr int voigt_index(int l, int m) noexcept {
    if (l == m)
        return l;
    if ((l == 0 && m == 1) || (l == 1 && m == 0))
        return 3;
    if ((l == 0 && m == 2) || (l == 2 && m == 0))
        return 4;
    return 5;
}

// --- Flat-array offset helpers ---

/// Offset into rmemory_sls[n_node × MEMORY_PER_NODE] for node, mechanism, voigt.
inline constexpr size_t sls_memory_offset(size_t node, int mechanism, int voigt) noexcept {
    return node * MEMORY_PER_NODE + static_cast<size_t>(mechanism) * VOIGT_COMPONENTS +
           static_cast<size_t>(voigt);
}

/// Offset into a relaxation-time array [n_node × N_SLS].
inline constexpr size_t tau_offset(size_t node, int mechanism) noexcept {
    return node * N_SLS + static_cast<size_t>(mechanism);
}

/// Offset into strain_old [n_node × VOIGT_COMPONENTS].
inline constexpr size_t strain_offset(size_t node, int component) noexcept {
    return node * VOIGT_COMPONENTS + static_cast<size_t>(component);
}

/// Offset into a per-mechanism coefficient [n_node × N_SLS].
inline constexpr size_t coef_offset(size_t node, int mechanism) noexcept {
    return node * N_SLS + static_cast<size_t>(mechanism);
}

/// Offset into forcing coefficients [n_node × N_SLS × FORCING_WEIGHTS].
inline constexpr size_t forcing_offset(size_t node, int mechanism, int time_level) noexcept {
    return (node * N_SLS + static_cast<size_t>(mechanism)) * FORCING_WEIGHTS +
           static_cast<size_t>(time_level);
}

// --- Coefficient precomputation ---

/// Compute exact piecewise-linear SLS recurrence coefficients for all GLL nodes.
///
/// @param tau_sigma      Per-node τ_σ array [n_node × N_SLS]
/// @param tau_epsilon_mu       Per-node shear τ_ε array [n_node × N_SLS]
/// @param tau_epsilon_kappa    Per-node bulk τ_ε array [n_node × N_SLS]
/// @param n_node         Number of GLL nodes on this rank
/// @param solver_dt      Simulation timestep (seconds)
/// @param[out] decay            exp(-dt/tau_sigma), [n_node × N_SLS]
/// @param[out] forcing_mu       previous/current deviatoric-strain weights
/// @param[out] forcing_kappa    previous/current volumetric-strain weights
void precompute_sls_coefficients(const double* tau_sigma, const double* tau_epsilon_mu,
                                 const double* tau_epsilon_kappa, int n_node, double solver_dt,
                                 double* decay, double* forcing_mu, double* forcing_kappa);

}  // namespace SLS

#endif  // GF_ATTENUATION_HPP_
