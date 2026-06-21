#pragma once

#include <vector>
#include <cstddef>
#include "gf/types.hpp"

namespace gf {

/// Perform MPI halo exchange for the residual/field array.
///
/// For each ExchangePattern in RankData:
///   1. Pack send_dof_indices from field into a contiguous send buffer
///   2. Post non-blocking send (MPI_Isend) to neighbor_rank
///   3. Post non-blocking receive (MPI_Irecv) from neighbor_rank into a recv buffer
///   4. MPI_Waitall on all requests
///   5. ADD received data into field at recv_dof_indices (accumulate, not overwrite)
///
/// The accumulation (add instead of overwrite) is essential for CG-SEM:
/// shared GLL node contributions from multiple neighbor ranks are summed.
///
/// \param patterns  Precomputed exchange patterns from RankData
/// \param field     Current field values [n_dof], modified in place (accumulate)
/// \param n_dof_per_node  DOF per GLL node (3 for displacement/velocity)
void exchange_halo(
    const std::vector<RankData::ExchangePattern>& patterns,
    std::vector<double>& field,
    int n_dof_per_node = 3
);

} // namespace gf