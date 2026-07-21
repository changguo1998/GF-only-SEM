/**
 * @file element_cpu.cpp
 * @brief Viscoelastic CPU element kernel stub — full implementation in Task 6.
 *
 * After Task 6, this file will call the five shared kernel_helpers and
 * the SLS viscoelastic stress computation.  For now it provides a
 * minimal implementation so the solver skeleton links.
 */

#define GF_ELEMENT_CPU_SOURCE
#include "gf/element.hpp"
#include "gf/kernel_helpers.hpp"

namespace gf {

template <>
void compute_element_residual<BackendCPU>(
    int n_elem, const double* dxi_dx, const double* jacobian,
    const double* lambda_, const double* mu_, const double* D,
    const double* weights, int NGLL, const double* u, double* r,
    const int32_t* /*pml_region*/, const double* /*pml_coef_strain*/,
    const double* /*rmemory_strain*/) {
    // Stub: zero-fill residual.  Full viscoelastic kernel in Task 6.
    const int n_node = NGLL * NGLL * NGLL;
    for (int e = 0; e < n_elem; ++e) {
        int offset = e * n_node * 3;
        for (int i = 0; i < n_node * 3; ++i) {
            r[offset + i] = 0.0;
        }
    }
    // Suppress unused parameter warnings
    (void)dxi_dx; (void)jacobian; (void)lambda_; (void)mu_;
    (void)D; (void)weights; (void)NGLL; (void)u;
}

}  // namespace gf