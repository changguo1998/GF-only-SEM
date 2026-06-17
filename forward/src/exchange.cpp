// forward/src/exchange.cpp
#include "gf/exchange.hpp"
#include <mpi.h>
#include <stdexcept>

namespace gf {

void exchange_halo(
    const std::vector<ExchangePattern>& patterns,
    const std::vector<double>& field,
    std::vector<double>& ghost_field,
    int n_dof_per_node
) {
    (void)n_dof_per_node; // reserved for validation if needed

    if (patterns.empty()) return;

    // Count total send/recv sizes for buffer allocation
    size_t total_send = 0;
    size_t total_recv = 0;
    for (const auto& p : patterns) {
        total_send += p.send_dof_indices.size();
        total_recv += p.recv_dof_indices.size();
    }

    // Allocate temporary buffers
    std::vector<double> send_buf(total_send);
    std::vector<double> recv_buf(total_recv);

    // MPI request arrays
    size_t n_patterns = patterns.size();
    std::vector<MPI_Request> requests(n_patterns * 2);

    // --- Pack and post non-blocking operations ---
    size_t send_offset = 0;
    size_t recv_offset = 0;

    for (size_t i = 0; i < n_patterns; ++i) {
        const auto& pat = patterns[i];
        size_t n_send = pat.send_dof_indices.size();
        size_t n_recv = pat.recv_dof_indices.size();

        // Pack send data
        for (size_t j = 0; j < n_send; ++j) {
            send_buf[send_offset + j] = field[pat.send_dof_indices[j]];
        }

        // Post non-blocking send
        MPI_Isend(send_buf.data() + send_offset,
                  static_cast<int>(n_send), MPI_DOUBLE,
                  pat.neighbor_rank, static_cast<int>(i),
                  MPI_COMM_WORLD,
                  &requests[2 * i]);

        // Post non-blocking receive
        MPI_Irecv(recv_buf.data() + recv_offset,
                  static_cast<int>(n_recv), MPI_DOUBLE,
                  pat.neighbor_rank, static_cast<int>(i),
                  MPI_COMM_WORLD,
                  &requests[2 * i + 1]);

        send_offset += n_send;
        recv_offset += n_recv;
    }

    // Wait for all communications to complete
    MPI_Waitall(static_cast<int>(n_patterns * 2), requests.data(), MPI_STATUSES_IGNORE);

    // --- Unpack received data into ghost_field ---
    recv_offset = 0;
    for (const auto& pat : patterns) {
        size_t n_recv = pat.recv_dof_indices.size();
        for (size_t j = 0; j < n_recv; ++j) {
            ghost_field[pat.recv_dof_indices[j]] = recv_buf[recv_offset + j];
        }
        recv_offset += n_recv;
    }
}

} // namespace gf