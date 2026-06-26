// forward/src/exchange.cpp
#include "gf/exchange.hpp"

#include <mpi.h>

#include <stdexcept>

#include "gf/types.hpp"

namespace gf {

void exchange_halo(const std::vector<RankData::ExchangePattern>& patterns,
                   std::vector<double>& field, int n_dof_per_node) {
    (void)n_dof_per_node;  // reserved for validation if needed

    if (patterns.empty())
        return;

    // Count total send/recv sizes for buffer allocation
    size_t total_send = 0;
    size_t total_recv = 0;
    for (const auto& p : patterns) {
        total_send += p.send_dof_indices.size();
        total_recv += p.recv_dof_indices.size();
    }

    if (total_send == 0 || total_recv == 0)
        return;

    // Allocate temporary buffers
    std::vector<double> send_buf(total_send);
    std::vector<double> recv_buf(total_recv);

    // MPI request arrays (init to MPI_REQUEST_NULL so skipped patterns are safe)
    size_t n_patterns = patterns.size();
    std::vector<MPI_Request> requests(n_patterns * 2, MPI_REQUEST_NULL);

    // --- Pack and post non-blocking operations ---
    size_t send_offset = 0;
    size_t recv_offset = 0;

    for (size_t i = 0; i < n_patterns; ++i) {
        const auto& pat = patterns[i];
        int n_send = static_cast<int>(pat.send_dof_indices.size());
        int n_recv = static_cast<int>(pat.recv_dof_indices.size());

        if (n_send == 0 || n_recv == 0)
            continue;

        // Pack send data
        for (int j = 0; j < n_send; ++j) {
            send_buf[send_offset + j] = field[pat.send_dof_indices[j]];
        }

        // Post non-blocking send
        constexpr int EXCHANGE_TAG = 42;
        MPI_Isend(send_buf.data() + send_offset, n_send, MPI_DOUBLE, pat.neighbor_rank,
                  EXCHANGE_TAG, MPI_COMM_WORLD, &requests[2 * i]);

        // Post non-blocking receive
        MPI_Irecv(recv_buf.data() + recv_offset, n_recv, MPI_DOUBLE, pat.neighbor_rank,
                  EXCHANGE_TAG, MPI_COMM_WORLD, &requests[2 * i + 1]);

        send_offset += n_send;
        recv_offset += n_recv;
    }

    // Wait for all communications to complete
    if (n_patterns > 0) {
        MPI_Waitall(static_cast<int>(n_patterns * 2), requests.data(), MPI_STATUSES_IGNORE);
    }

    // --- Accumulate received data into field (add, not overwrite) ---
    // This is the key CG-SEM assembly step: contributions from neighbor
    // ranks at shared GLL nodes are summed into the local residual.
    recv_offset = 0;
    for (const auto& pat : patterns) {
        int n_recv = static_cast<int>(pat.recv_dof_indices.size());
        for (int j = 0; j < n_recv; ++j) {
            field[pat.recv_dof_indices[j]] += recv_buf[recv_offset + j];
        }
        recv_offset += n_recv;
    }
}

}  // namespace gf