/**
 * @file attenuation.cpp
 * @brief SLS viscoelastic attenuation — precomputation and memory update.
 */

#include "gf/attenuation.hpp"

namespace SLS {

void precompute_sls_coefficients(const double* tau_sigma, const double* tau_epsilon_mu,
                                 const double* tau_epsilon_kappa, int n_node, double solver_dt,
                                 double* decay, double* forcing_mu, double* forcing_kappa) {
    for (int node = 0; node < n_node; ++node) {
        double ratio_sum_mu = 0.0;
        double ratio_sum_kappa = 0.0;
        for (int l = 0; l < N_SLS; ++l) {
            const size_t offset = tau_offset(node, l);
            ratio_sum_mu += tau_epsilon_mu[offset] / tau_sigma[offset];
            ratio_sum_kappa += tau_epsilon_kappa[offset] / tau_sigma[offset];
        }

        for (int l = 0; l < N_SLS; ++l) {
            const size_t offset = tau_offset(node, l);
            const double tau_s = tau_sigma[offset];
            const double step_ratio = solver_dt / tau_s;
            const double a = std::exp(-step_ratio);
            const double previous_time_weight = (1.0 - a) / step_ratio - a;
            const double current_time_weight = 1.0 - (1.0 - a) / step_ratio;

            const double weight_mu = (tau_epsilon_mu[offset] / tau_s - 1.0) / ratio_sum_mu;
            const double weight_kappa =
                (tau_epsilon_kappa[offset] / tau_s - 1.0) / ratio_sum_kappa;

            decay[coef_offset(node, l)] = a;
            forcing_mu[forcing_offset(node, l, 0)] = weight_mu * previous_time_weight;
            forcing_mu[forcing_offset(node, l, 1)] = weight_mu * current_time_weight;
            forcing_kappa[forcing_offset(node, l, 0)] = weight_kappa * previous_time_weight;
            forcing_kappa[forcing_offset(node, l, 1)] = weight_kappa * current_time_weight;
        }
    }
}

}  // namespace SLS
