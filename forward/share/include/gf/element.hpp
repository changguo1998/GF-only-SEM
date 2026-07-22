#pragma once

#include <cstddef>
#include <vector>

#include "gf/backend.hpp"
#include "gf/types.hpp"

namespace gf {

// -----------------------------------------------------------------------
// compute_element_residual — backend-dispatched kernel
//
// Compute the internal force (stiffness residual) for a batch of elements.
// Matrix-free: no global stiffness matrix.
//
// All arrays are element-major contiguous (n_elem blocks of NGLL^3 nodes).
//
// C-PML and SLS attenuation parameters have nullptr defaults — callers
// that don't need them can omit the extra arguments entirely.
// -----------------------------------------------------------------------

template <typename Backend>
void compute_element_residual(int n_elem, const double* dxi_dx, const double* jacobian,
                              const double* lambda_, const double* mu_, const double* D,
                              const double* weights, int NGLL, const double* u, double* r,
                              const int32_t* pml_region = nullptr,
                              const double* pml_coef_strain = nullptr,
                              const double* rmemory_strain = nullptr,
                              double* rmemory_sls = nullptr, double* sigma_old = nullptr,
                              const double* sls_coef_a = nullptr,
                              const double* sls_coef_b = nullptr, bool has_attenuation = false);

// --- Explicit instantiation declarations ---
// Guarded: specialization source files define GF_ELEMENT_{CPU,CUDA}_SOURCE
// to suppress the extern template declaration and avoid
// "specialization after instantiation" errors.

#ifndef GF_ELEMENT_CPU_SOURCE
extern template void compute_element_residual<BackendCPU>(
    int, const double*, const double*, const double*, const double*, const double*, const double*,
    int, const double*, double*, const int32_t*, const double*, const double*, double*, double*,
    const double*, const double*, bool);
#endif

#ifdef GF_WITH_CUDA
#ifndef GF_ELEMENT_CUDA_SOURCE
extern template void compute_element_residual<BackendCUDA>(
    int, const double*, const double*, const double*, const double*, const double*, const double*,
    int, const double*, double*, const int32_t*, const double*, const double*, double*, double*,
    const double*, const double*, bool);
#endif
#endif

}  // namespace gf