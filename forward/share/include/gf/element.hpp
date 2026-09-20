#pragma once

#include <cstddef>
#include <vector>

#include "gf/backend.hpp"
#include "gf/types.hpp"

namespace gf {

// -----------------------------------------------------------------------
// compute_element_residual — element-stiffness (internal-force) kernel
//
// Compute the internal force (stiffness residual) for a batch of elements.
// Matrix-free: no global stiffness matrix.
//
// All arrays are element-major contiguous (n_elem blocks of NGLL^3 nodes).
//
// C-PML and SLS attenuation parameters have nullptr defaults — callers
// that don't need them can omit the extra arguments entirely.
//
// The function name is the same for every physics variant (elastic,
// viscoelastic, ...). The correct implementation is selected at link
// time by the library linked (libgf_elastic, libgf_visco, ...).
// -----------------------------------------------------------------------

void compute_element_residual(
    int n_elem, const double* dxi_dx, const double* jacobian, const double* lambda_,
    const double* mu_, const double* D, const double* weights, int NGLL, const double* u,
    double* r, const int32_t* pml_region = nullptr, const double* pml_coef_strain = nullptr,
    const double* rmemory_strain = nullptr, double* rmemory_sls = nullptr,
    double* strain_old = nullptr, const double* sls_decay = nullptr,
    const double* sls_forcing_mu = nullptr, const double* sls_forcing_kappa = nullptr,
    bool has_attenuation = false);

}  // namespace gf
