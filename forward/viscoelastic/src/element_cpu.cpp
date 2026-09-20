/**
 * @file element_cpu.cpp
 * @brief Viscoelastic CPU element kernel with SLS attenuation.
 *
 * Calls the five shared geometry/mechanics helpers from kernel_helpers.hpp.
 * Replaces the elastic isotropic stress with SLS viscoelastic stress:
 *   sigma = sigma_elastic - sum(R_l)
 * and updates the SLS memory variables inline.
 */

#include "gf/attenuation.hpp"
#include "gf/element.hpp"
#include "gf/kernel_helpers.hpp"
#include "gf/types.hpp"

namespace gf {

namespace {
/// 1D flat index from (i, j, k) within an element.
inline int idx(int i, int j, int k, int NGLL) {
    return (i * NGLL + j) * NGLL + k;
}
}  // anonymous namespace

#ifndef GF_WITH_CUDA
void compute_element_residual(int n_elem, const double* dxi_dx, const double* jacobian,
                              const double* lambda_, const double* mu_, const double* D,
                              const double* weights, int NGLL, const double* u, double* r,
                              const int32_t* pml_region, const double* pml_coef_strain,
                              const double* rmemory_strain, double* rmemory_sls,
                              double* strain_old, const double* sls_decay,
                              const double* sls_forcing_mu, const double* sls_forcing_kappa,
                              bool has_attenuation) {
    const int n_node = NGLL * NGLL * NGLL;

    for (int elem = 0; elem < n_elem; ++elem) {
        const double* elem_dxi_dx = dxi_dx + elem * n_node * 9;
        const double* elem_jac = jacobian + elem * n_node;
        const double* elem_lambda = lambda_ + elem * n_node;
        const double* elem_mu = mu_ + elem * n_node;
        const double* elem_u = u + elem * n_node * 3;
        double* elem_r = r + elem * n_node * 3;

        for (int i = 0; i < NGLL; ++i) {
            for (int j = 0; j < NGLL; ++j) {
                for (int k = 0; k < NGLL; ++k) {
                    const int n = idx(i, j, k, NGLL);
                    const int global_node = elem * n_node + n;

                    const double lambda = elem_lambda[n];
                    const double mu = elem_mu[n];
                    if (mu <= 0.0)
                        continue;

                    const double* dd = &elem_dxi_dx[9 * n];

                    // [1] Reference-space gradient
                    double dudxi[3], dudeta[3], dudzeta[3];
                    compute_reference_gradient(i, j, k, NGLL, D, elem_u, dudxi, dudeta, dudzeta);

                    // [2] Physical gradient
                    double du_dx[3][3];
                    transform_to_physical(dudxi, dudeta, dudzeta, dd, du_dx);

                    // [3-5] PML non-symmetric stress or symmetric SLS path
                    double sigma[3][3];
                    if (pml_region && pml_region[elem] != 0) {
                        compute_pml_non_symmetric_stress(global_node, du_dx, lambda, mu,
                                                         pml_coef_strain, rmemory_strain, sigma);
                    } else {
                        // [4] Strain tensor
                        double eps[3][3];
                        compute_strain_tensor(du_dx, eps);

                        // === Step A: elastic trial stress ===
                        const double eps_kk = eps[0][0] + eps[1][1] + eps[2][2];
                        for (int l = 0; l < 3; ++l) {
                            for (int m = 0; m < 3; ++m) {
                                sigma[l][m] = 2.0 * mu * eps[l][m];
                            }
                            sigma[l][l] += lambda * eps_kk;
                        }

                        // === Step B: subtract old deviatoric and bulk memory ===
                        if (has_attenuation && rmemory_sls != nullptr) {
                            const double dev_xx = eps[0][0] - eps_kk / 3.0;
                            const double dev_yy = eps[1][1] - eps_kk / 3.0;
                            const double current_strain[SLS::VOIGT_COMPONENTS] = {
                                dev_xx, dev_yy, eps_kk, eps[0][1], eps[0][2], eps[1][2]};

                            double memory_sum[SLS::VOIGT_COMPONENTS] = {};
                            for (int mechanism = 0; mechanism < SLS::N_SLS; ++mechanism) {
                                for (int component = 0; component < SLS::VOIGT_COMPONENTS;
                                     ++component) {
                                    memory_sum[component] += rmemory_sls[SLS::sls_memory_offset(
                                        global_node, mechanism, component)];
                                }
                            }
                            sigma[0][0] -= memory_sum[SLS::DEVIATORIC_XX] + memory_sum[SLS::TRACE];
                            sigma[1][1] -= memory_sum[SLS::DEVIATORIC_YY] + memory_sum[SLS::TRACE];
                            sigma[2][2] += memory_sum[SLS::DEVIATORIC_XX] +
                                           memory_sum[SLS::DEVIATORIC_YY] - memory_sum[SLS::TRACE];
                            sigma[0][1] -= memory_sum[SLS::XY];
                            sigma[0][2] -= memory_sum[SLS::XZ];
                            sigma[1][2] -= memory_sum[SLS::YZ];

                            // === Step C: update memory from previous/current strain ===
                            const double kappa = lambda + 2.0 * mu / 3.0;
                            for (int mechanism = 0; mechanism < SLS::N_SLS; ++mechanism) {
                                const double decay =
                                    sls_decay[SLS::coef_offset(global_node, mechanism)];
                                const double mu_previous =
                                    sls_forcing_mu[SLS::forcing_offset(global_node, mechanism, 0)];
                                const double mu_current =
                                    sls_forcing_mu[SLS::forcing_offset(global_node, mechanism, 1)];
                                const double kappa_previous =
                                    sls_forcing_kappa[SLS::forcing_offset(global_node, mechanism,
                                                                          0)];
                                const double kappa_current = sls_forcing_kappa[SLS::forcing_offset(
                                    global_node, mechanism, 1)];

                                for (int component = SLS::DEVIATORIC_XX;
                                     component <= SLS::DEVIATORIC_YY; ++component) {
                                    const size_t memory_offset =
                                        SLS::sls_memory_offset(global_node, mechanism, component);
                                    const double previous =
                                        strain_old[SLS::strain_offset(global_node, component)];
                                    rmemory_sls[memory_offset] =
                                        decay * rmemory_sls[memory_offset] +
                                        2.0 * mu *
                                            (mu_previous * previous +
                                             mu_current * current_strain[component]);
                                }
                                for (int component = SLS::XY; component <= SLS::YZ; ++component) {
                                    const size_t memory_offset =
                                        SLS::sls_memory_offset(global_node, mechanism, component);
                                    const double previous =
                                        strain_old[SLS::strain_offset(global_node, component)];
                                    rmemory_sls[memory_offset] =
                                        decay * rmemory_sls[memory_offset] +
                                        2.0 * mu *
                                            (mu_previous * previous +
                                             mu_current * current_strain[component]);
                                }
                                const size_t bulk_offset =
                                    SLS::sls_memory_offset(global_node, mechanism, SLS::TRACE);
                                const double previous_trace =
                                    strain_old[SLS::strain_offset(global_node, SLS::TRACE)];
                                rmemory_sls[bulk_offset] =
                                    decay * rmemory_sls[bulk_offset] +
                                    kappa *
                                        (kappa_previous * previous_trace + kappa_current * eps_kk);
                            }

                            for (int component = 0; component < SLS::VOIGT_COMPONENTS;
                                 ++component) {
                                strain_old[SLS::strain_offset(global_node, component)] =
                                    current_strain[component];
                            }
                        }

                        // Symmetrize
                        sigma[1][0] = sigma[0][1];
                        sigma[2][0] = sigma[0][2];
                        sigma[2][1] = sigma[1][2];
                    }  // end SLS (non-PML) path

                    // [5] Residual scatter
                    scatter_residual(i, j, k, NGLL, sigma, dd, D, weights, elem_jac[n], elem_r);
                }
            }
        }
    }
}

#endif  // !GF_WITH_CUDA
}  // namespace gf
