// tests/test_exchange.cpp — MPI halo exchange tests
//
// NOTE: This test requires mpirun with at least 2 ranks:
//   mpirun -np 2 ./tests/test_exchange
//
// Single-rank runs skip the multi-rank tests gracefully.
#define CATCH_CONFIG_RUNNER
#include <mpi.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "gf/exchange.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}

// Helper: build patterns for two-rank exchange
// Rank 0 sends DOFs 0-5 to rank 1; rank 1 receives into DOFs 0-5.
static std::vector<RankData::ExchangePattern> make_pairwise_patterns(int rank, int nprocs) {
    std::vector<RankData::ExchangePattern> patterns;

    if (nprocs < 2)
        return patterns;

    RankData::ExchangePattern pat;
    if (rank == 0) {
        pat.neighbor_rank = 1;
        // Rank 0 sends its own DOFs 0..5 to rank 1
        pat.send_dof_indices = {0, 1, 2, 3, 4, 5};
        // Rank 0 receives rank 1's data into ghost DOFs 6..11
        pat.recv_dof_indices = {6, 7, 8, 9, 10, 11};
    } else if (rank == 1) {
        pat.neighbor_rank = 0;
        // Rank 1 sends its own DOFs 0..5 to rank 0
        pat.send_dof_indices = {0, 1, 2, 3, 4, 5};
        // Rank 1 receives rank 0's data into ghost DOFs 6..11
        pat.recv_dof_indices = {6, 7, 8, 9, 10, 11};
    }
    patterns.push_back(pat);
    return patterns;
}

TEST_CASE("Exchange halo transfers data between two ranks", "[exchange][mpi]") {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2) {
        SKIP("This test requires at least 2 MPI ranks");
    }

    auto patterns = make_pairwise_patterns(rank, nprocs);
    REQUIRE(patterns.size() == 1);

    // Each rank has a field of length 12 (6 owned + 6 ghost)
    std::vector<double> field(12, 0.0);

    // Rank 0 fills DOFs 0..5 with [10, 11, 12, 13, 14, 15]
    // Rank 1 fills DOFs 0..5 with [20, 21, 22, 23, 24, 25]
    if (rank == 0) {
        for (int i = 0; i < 6; ++i)
            field[i] = 10.0 + i;
    } else {
        for (int i = 0; i < 6; ++i)
            field[i] = 20.0 + i;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Perform exchange
    exchange_halo(patterns, field, 1);

    MPI_Barrier(MPI_COMM_WORLD);

    // After exchange:
    // Rank 0: owned DOFs [10..15] unchanged, ghost DOFs [6..11] = rank 1's [20..25]
    // Rank 1: owned DOFs [20..25] unchanged, ghost DOFs [6..11] = rank 0's [10..15]
    if (rank == 0) {
        for (int i = 0; i < 6; ++i) {
            REQUIRE(field[i] == 10.0 + i);
            REQUIRE(field[i + 6] == 20.0 + i);
        }
    } else {
        for (int i = 0; i < 6; ++i) {
            REQUIRE(field[i] == 20.0 + i);
            REQUIRE(field[i + 6] == 10.0 + i);
        }
    }
}

TEST_CASE("Exchange accumulate adds received data (CG-SEM assembly)", "[exchange][mpi]") {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs < 2) {
        SKIP("This test requires at least 2 MPI ranks");
    }

    // Build patterns where both ranks send to the same recv indices
    // Rank 0 receives into DOFs 0..5 (overwriting own sent data area)
    // Rank 1 receives into DOFs 0..5
    std::vector<RankData::ExchangePattern> patterns;
    RankData::ExchangePattern pat;
    if (rank == 0) {
        pat.neighbor_rank = 1;
        pat.send_dof_indices = {0, 1, 2};
        pat.recv_dof_indices = {0, 1, 2};  // accumulate, not overwrite
    } else if (rank == 1) {
        pat.neighbor_rank = 0;
        pat.send_dof_indices = {0, 1, 2};
        pat.recv_dof_indices = {0, 1, 2};
    }
    patterns.push_back(pat);

    // Each rank has field = [1.0, 2.0, 3.0, ...]
    std::vector<double> field(6, 0.0);
    for (int i = 0; i < 6; ++i)
        field[i] = rank * 10.0 + i + 1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    exchange_halo(patterns, field, 1);
    MPI_Barrier(MPI_COMM_WORLD);

    // After accumulation:
    // Rank 0: field[0..2] = own + rank1's = [1,2,3] + [11,12,13] = [12,14,16]
    // Rank 1: field[0..2] = own + rank0's = [11,12,13] + [1,2,3] = [12,14,16]
    // Both ranks should have identical values at shared nodes after assembly
    double expected0 = 1.0 + 11.0;  // 12
    double expected1 = 2.0 + 12.0;  // 14
    double expected2 = 3.0 + 13.0;  // 16

    REQUIRE(field[0] == expected0);
    REQUIRE(field[1] == expected1);
    REQUIRE(field[2] == expected2);

    // DOFs 3..5 should be unchanged
    double own3 = rank * 10.0 + 4.0;
    REQUIRE(field[3] == own3);
}

TEST_CASE("Exchange with empty patterns is no-op", "[exchange]") {
    std::vector<RankData::ExchangePattern> empty;
    std::vector<double> field = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Should not crash or modify field
    exchange_halo(empty, field, 1);

    REQUIRE(field[0] == 1.0);
    REQUIRE(field[1] == 2.0);
    REQUIRE(field[2] == 3.0);
    REQUIRE(field[3] == 4.0);
    REQUIRE(field[4] == 5.0);
}