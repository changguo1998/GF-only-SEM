#define GF_ELEMENT_CPU_SOURCE
#include <cmath>

#include "gf/element.hpp"
#include "gf/gll.hpp"
#include "gf/kernel_helpers.hpp"
#include "gf/pml.hpp"

namespace gf {

// Inline helper: 1D flat index from (i, j, k) in element
static inline int idx(int i, int j, int k, int NGLL) {
    return (i * NGLL + j) * NGLL + k;
}

/// Compute element residual for a batch of elements on CPU — elastic.
///
/// Calls the five shared geometry/mechanics helpers from kernel_helpers.hpp
/// and inserts the elastic isotropic stress law between strain and scatter.
template <>
void compute_element_residual<BackendCPU>(int n_elem, const double* dxi_dx, const double* jacobian,
                                          const double* lambda_, const double* mu_,
                                          const double* D, const double* weights, int NGLL,
                                          const double* u, double* r, const int32_t* pml_region,
                                          const double* pml_coef_strain,
                                          const double* rmemory_strain, double* /*rmemory_sls*/,
                                          double* /*sigma_old*/, const double* /*sls_coef_a*/,
                                          const double* /*sls_coef_b*/, bool /*has_attenuation*/) {
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

                    // --- Material coefficients ---
                    const double lambda = elem_lambda[n];
                    const double mu = elem_mu[n];
                    if (mu <= 0.0)
                        continue;

                    const double* dd = &elem_dxi_dx[9 * n];

                    // --- [1] Reference-space gradient ---
                    double dudxi[3], dudeta[3], dudzeta[3];
                    compute_reference_gradient(i, j, k, NGLL, D, elem_u, dudxi, dudeta, dudzeta);

                    // --- [2] Physical gradient ---
                    double du_dx[3][3];
                    transform_to_physical(dudxi, dudeta, dudzeta, dd, du_dx);

                    // --- [3-5] Stress and scatter ---
                    double sigma[3][3];
                    if (pml_region && pml_region[elem] != 0) {
                        const int global_node = elem * n_node + n;
                        compute_pml_non_symmetric_stress(global_node, du_dx, lambda, mu,
                                                         pml_coef_strain, rmemory_strain, sigma);
                    } else {
                        double eps[3][3];
                        compute_strain_tensor(du_dx, eps);
                        double eps_kk = eps[0][0] + eps[1][1] + eps[2][2];
                        for (int l = 0; l < 3; ++l) {
                            for (int m = 0; m < 3; ++m) {
                                sigma[l][m] = 2.0 * mu * eps[l][m];
                            }
                            sigma[l][l] += lambda * eps_kk;
                        }
                    }
                    scatter_residual(i, j, k, NGLL, sigma, dd, D, weights, elem_jac[n], elem_r);
                }
            }
        }
    }
}

}  // namespace gf