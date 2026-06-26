#pragma once

#include <cstddef>
#include <vector>

#include "gf/types.hpp"

namespace gf {

/// Newmark explicit predictor step (beta=0, gamma=0.5).
///
/// Computes predicted displacement and velocity at t_{n+1}:
///   u_tilde = u + dt * v + 0.5 * dt^2 * a
///   v_tilde = v + 0.5 * dt * a
///
/// @param[in]  params    Newmark parameters (beta=0, gamma=0.5, dt)
/// @param[in]  u         displacement at t_n [n_dof]
/// @param[in]  v         velocity at t_n [n_dof]
/// @param[in]  a         acceleration at t_n [n_dof]
/// @param[out] u_tilde   predicted displacement at t_{n+1} [n_dof]
/// @param[out] v_tilde   predicted velocity at t_{n+1} [n_dof]
void newmark_predictor(const NewmarkParams& params, const std::vector<double>& u,
                       const std::vector<double>& v, const std::vector<double>& a,
                       std::vector<double>& u_tilde, std::vector<double>& v_tilde);

/// Newmark explicit corrector step (beta=0, gamma=0.5).
///
/// Computes acceleration at t_{n+1} and corrects velocity:
///   a_new[i] = residual[i] / mass[i]
///   v_new = v_tilde + 0.5 * dt * a_new
///   u unchanged (beta=0 means no displacement correction)
///
/// @param[in]  params   Newmark parameters
/// @param[in]  mass     lumped mass diagonal [n_dof]
/// @param[in]  residual residual force vector at predicted state [n_dof]
/// @param[in,out] u     displacement (unchanged when beta=0)
/// @param[in,out] v     velocity, updated to v_{n+1}
/// @param[out] a        acceleration, computed as r/mass
void newmark_corrector(const NewmarkParams& params, const std::vector<double>& mass,
                       const std::vector<double>& residual, std::vector<double>& u,
                       std::vector<double>& v, std::vector<double>& a);

}  // namespace gf