/**
 * @file solver.cpp
 * @brief Viscoelastic SEM solver stub — full implementation in Task 7.
 */

#include "gf/solver_viscoelastic.hpp"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace gf {

int run_viscoelastic_forward(const std::string& direction, bool /*resume_mode*/,
                             int /*effective_nprocs*/) {
    // Stub: full solver integration is implemented in Task 7.
    printf("[viscoelastic solver] direction=%s — stub: not yet implemented.\n",
           direction.c_str());
    return 0;
}

}  // namespace gf