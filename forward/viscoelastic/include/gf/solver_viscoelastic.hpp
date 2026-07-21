/**
 * @file solver_viscoelastic.hpp
 * @brief Viscoelastic SEM forward solver with SLS attenuation.
 */

#pragma once

#include <string>

namespace gf {

/// Run the viscoelastic forward simulation for a single source direction.
///
/// @param direction            Force direction: "x", "y", or "z"
/// @param resume_mode          If true, load restart files
/// @param effective_nprocs     GPU reduction mode: 0=auto, >0=reduce
/// @return 0 on success, 1 on error
int run_viscoelastic_forward(const std::string& direction, bool resume_mode,
                             int effective_nprocs = 0);

}  // namespace gf