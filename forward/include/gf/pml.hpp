#pragma once

#include <vector>
#include <cstddef>
#include "gf/types.hpp"

namespace gf {

/// Apply precomputed PML damping to the velocity field.
///
/// For each DOF, the velocity is damped by a precomputed damping profile:
///   v[i] -= damping_profile[node] * v[i]
///
/// The damping_profile is a per-GLL-node array (length n_elem * NGLL^3).
/// All 3 DOF components at a node share the same damping coefficient.
/// damping_profile[node] = 0 for interior nodes (no damping).
///
/// @param[in] damping_profile  per-GLL-node damping values [n_elem * NGLL^3]
/// @param[in] u                displacement field [n_dof]
/// @param[in,out] v            velocity field [n_dof], modified in place
/// @param[in] n_dof            total DOF count (= n_elem * NGLL^3 * 3)
void apply_pml_damping(
    const std::vector<double>& damping_profile,
    const std::vector<double>& u,
    std::vector<double>& v,
    int n_dof
);

} // namespace gf