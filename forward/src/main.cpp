// forward/src/main.cpp
//
// gf_solver --direction {x,y,z}
// All I/O paths are frozen relative to CWD:
//   Input:  config.h5, partitions/partition_{r}.h5
//   Output: wavefields/{direction}/record_{r}.h5

#include "gf/solver.hpp"
#include <mpi.h>
#include <iostream>
#include <string>
#include <cstring>
#include <stdexcept>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --direction {x,y,z}\n"
              << "  All I/O paths are frozen relative to CWD:\n"
              << "    Input:  config.h5, partitions/partition_{r}.h5\n"
              << "    Output: wavefields/{direction}/record_{r}.h5\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        std::string direction;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--direction") == 0 && i + 1 < argc) {
                direction = argv[++i];
            }
        }

        if (direction != "x" && direction != "y" && direction != "z") {
            if (rank == 0) {
                std::cerr << "Error: --direction must be x, y, or z, got '"
                          << direction << "'\n";
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }

        if (rank == 0) {
            std::cout << "gf_solver: direction=" << direction << std::endl;
            std::cout << "  input:  config.h5 + partitions/partition_{r}.h5\n"
                      << "  output: wavefields/" << direction << "/record_{r}.h5" << std::endl;
        }

        int result = gf::run_forward(direction);

        MPI_Finalize();
        return result;

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Fatal error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
}