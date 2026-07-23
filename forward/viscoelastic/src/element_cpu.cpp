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

// Voigt component -> tensor index pairs
static constexpr int VMAP[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};

#ifndef GF_WITH_CUDA
void compute_element_residual(int n_elem, const double* dxi_dx, const double* jacobian,
                              const double* lambda_, const double* mu_, const double* D,
                              const double* weights, int NGLL, const double* u, double* r,
                              const int32_t* pml_region, const double* pml_coef_strain,
                              const double* rmemory_strain, double* rmemory_sls, double* sigma_old,
                              const double* sls_coef_a, const double* sls_coef_b,
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
                        double eps_kk = eps[0][0] + eps[1][1] + eps[2][2];
                        double sigma[3][3];
                        for (int l = 0; l < 3; ++l) {
                            for (int m = 0; m < 3; ++m) {
                                sigma[l][m] = 2.0 * mu * eps[l][m];
                            }
                            sigma[l][l] += lambda * eps_kk;
                        }

                        // === Step B: SLS memory subtraction + update ===
                        if (has_attenuation && rmemory_sls != nullptr) {
                            for (int sls = 0; sls < SLS::N_SLS; ++sls) {
                                double a = sls_coef_a[SLS::coef_offset(global_node, sls)];
                                double b = sls_coef_b[SLS::coef_offset(global_node, sls)];

                                for (int v = 0; v < SLS::VOIGT_COMPONENTS; ++v) {
                                    int l = VMAP[v][0];
                                    int m = VMAP[v][1];

                                    double sigma_prev =
                                        sigma_old[SLS::sigma_old_offset(global_node, v)];

                                    size_t mem_off = SLS::sls_memory_offset(global_node, sls, v);

                                    // R = a * R + b * (sigma_curr - sigma_prev)
                                    double delta = sigma[l][m] - sigma_prev;
                                    rmemory_sls[mem_off] = a * rmemory_sls[mem_off] + b * delta;

                                    // Subtract memory from elastic stress
                                    sigma[l][m] -= rmemory_sls[mem_off];
                                }
                            }

                            // Save current stress for next timestep
                            for (int v = 0; v < SLS::VOIGT_COMPONENTS; ++v) {
                                int l = VMAP[v][0];
                                int m = VMAP[v][1];
                                sigma_old[SLS::sigma_old_offset(global_node, v)] = sigma[l][m];
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