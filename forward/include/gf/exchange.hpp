#pragma once

#include <vector>
#include <cstddef>

namespace gf {

/// Description of one neighbor face for MPI halo exchange.
/// Precomputed during preprocessing from partition topology.
struct ExchangePattern {
    int neighbor_rank;                    // MPI rank of the neighbor
    std::vector<int> send_node_indices;   // local element GLL node indices to send
    std::vector<int> send_dof_indices;    // local DOF indices (= 3*node_idx + dir) to pack
    std::vector<int> recv_node_indices;   // ghost element GLL node indices to fill
    std::vector<int> recv_dof_indices;    // ghost DOF indices to write into ghost_field
};

/// Perform MPI halo exchange for the displacement/velocity field.
///
/// For each ExchangePattern:
///   1. Pack send_dof_indices from field into a contiguous send buffer
///   2. Post non-blocking send (MPI_Isend) to neighbor_rank
///   3. Post non-blocking receive (MPI_Irecv) from neighbor_rank into a recv buffer
///   4. MPI_Waitall on all requests
///   5. Unpack recv buffer into ghost_field at recv_dof_indices
///
/// \param patterns  Precomputed exchange patterns for this rank
/// \param field     Current field values [n_dof], owned by this rank
/// \param ghost_field  Output ghost field [n_ghost_dof], written by this function
/// \param n_dof_per_node  DOF per GLL node (3 for displacement/velocity)
void exchange_halo(
    const std::vector<ExchangePattern>& patterns,
    const std::vector<double>& field,
    std::vector<double>& ghost_field,
    int n_dof_per_node = 3
);

} // namespace gf