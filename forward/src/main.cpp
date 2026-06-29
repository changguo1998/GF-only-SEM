// forward/src/main.cpp
//
// gf_solver --direction {x,y,z} [--resume]
// All I/O paths are frozen relative to CWD:
//   Input:  config.h5, partitions/partition_{r}.h5
//   Output: wavefields/{direction}/record_{r}.h5
//   Restart: restart/{direction}/restart_{r}.h5 (with --resume)

#ifndef GF_NO_MPI
#include <mpi.h>
#endif

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gf/solver.hpp"

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " --direction {x,y,z} [--resume]\n"
              << "  All I/O paths are frozen relative to CWD:\n"
              << "    Input:  config.h5, partitions/partition_{r}.h5\n"
              << "    Output: wavefields/{direction}/record_{r}.h5\n"
              << "    Restart: restart/{direction}/restart_{r}.h5\n";
}

int main(int argc, char** argv) {
#ifndef GF_NO_MPI
    MPI_Init(&argc, &argv);
#endif
    int rank = 0;
#ifndef GF_NO_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    try {
        std::string direction;
        bool resume_mode = false;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--direction") == 0 && i + 1 < argc) {
                direction = argv[++i];
            } else if (std::strcmp(argv[i], "--resume") == 0) {
                resume_mode = true;
            }
        }

        if (direction != "x" && direction != "y" && direction != "z") {
            if (rank == 0) {
                std::cerr << "Error: --direction must be x, y, or z, got '" << direction << "'\n";
                print_usage(argv[0]);
            }
#ifndef GF_NO_MPI
            MPI_Finalize();
#endif
            return 1;
        }

        if (rank == 0) {
            std::cout << "gf_solver: direction=" << direction;
            if (resume_mode)
                std::cout << " (resume mode)";
            std::cout << std::endl;
            std::cout << "  input:  config.h5 + partitions/partition_{r}.h5\n"
                      << "  output: wavefields/" << direction << "/record_{r}.h5" << std::endl;
        }

        int result = gf::run_forward(direction, resume_mode);

#ifndef GF_NO_MPI
        MPI_Finalize();
#endif
        return result;

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Fatal error: " << e.what() << std::endl;
#ifndef GF_NO_MPI
        MPI_Finalize();
#endif
        return 1;
    }
}