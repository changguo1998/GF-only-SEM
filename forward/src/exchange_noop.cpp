// forward/src/exchange_noop.cpp
//
// No-op MPI halo exchange for single-process (non-MPI) builds.
// Used by gf_solver_cuda (single-GPU, no MPI).

#include "gf/exchange.hpp"

namespace gf {

void exchange_halo(const std::vector<RankData::ExchangePattern>& /*patterns*/,
                   std::vector<double>& /*field*/, int /*n_dof_per_node*/) {
    // No-op: single process has no neighbors to exchange with.
}

}  // namespace gf