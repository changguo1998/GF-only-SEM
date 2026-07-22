/**
 * @file attenuation.cpp
 * @brief SLS viscoelastic attenuation — precomputation and memory update.
 */

#include "gf/attenuation.hpp"

namespace SLS {

void precompute_sls_coefficients(const double* tau_sigma, const double* tau_epsilon, int n_node,
                                 double solver_dt, double* coef_a, double* coef_b) {
    for (int node = 0; node < n_node; ++node) {
        for (int l = 0; l < N_SLS; ++l) {
            double tau_s = tau_sigma[tau_offset(node, l)];
            double tau_e = tau_epsilon[tau_offset(node, l)];

            // Guard: tau_s must be positive and > solver_dt for numerical stability.
            // When Q → ∞ (no attenuation), tau_e ≈ tau_s → b ≈ 0, R never accumulates.
            double a = std::exp(-solver_dt / tau_s);
            double ratio = tau_e / tau_s;
            double b = (ratio - 1.0) * (1.0 - a);

            coef_a[coef_offset(node, l)] = a;
            coef_b[coef_offset(node, l)] = b;
        }
    }
}

}  // namespace SLS