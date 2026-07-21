/**
 * @file solver.cpp
 * @brief Viscoelastic SEM forward solver entry point.
 *
 * Delegates to the shared run_forward() in libgf_shared.  The viscoelastic
 * behaviour comes from linking the viscoelastic element kernel (libgf_visco)
 * instead of the elastic one (libgf).  The shared solver picks up the
 * correct specialization of compute_element_residual<ActiveBackend> at
 * link time.
 *
 * The model.h5 file must contain /field/cell/tau_sigma and
 * /field/cell/tau_epsilon datasets (written by preprocess/attenuation.py).
 * When these are present, io.cpp sets part.has_attenuation = true and the
 * shared solver automatically enables SLS memory allocation, coefficient
 * precomputation, and passes SLS parameters to the element kernel.
 */

#include "gf/solver.hpp"       // run_forward()
#include "gf/solver_viscoelastic.hpp"

#include <cstdio>

namespace gf {

int run_viscoelastic_forward(const std::string& direction, bool resume_mode,
                             int effective_nprocs) {
    // The shared run_forward() handles everything:
    //   - Read model / config / partitions
    //   - Setup GLL matrices and Newmark parameters
    //   - Detect has_attenuation from model.h5 tau_sigma dataset
    //   - Precompute SLS coefficients and allocate memory
    //   - Time loop with SLS-aware compute_element_residual
    //   - Snapshot and restart I/O
    //
    // The viscoelastic element kernel is used because this binary is linked
    // against libgf_visco (which provides the SLS specialization of
    // compute_element_residual<BackendCPU>).

    return run_forward(direction, resume_mode, effective_nprocs);
}

}  // namespace gf