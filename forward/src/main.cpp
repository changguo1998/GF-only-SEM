// forward/src/main.cpp
#include "gf/solver.hpp"
#include <mpi.h>
#include <iostream>
#include <string>
#include <cstring>
#include <stdexcept>

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <partition_dir> <config.h5> <output_dir> --direction {x,y,z}\n"
              << "  partition_dir: directory containing partition_{0..N-1}.h5\n"
              << "  config.h5:     rank-invariant simulation config\n"
              << "  output_dir:    output directory for record files\n"
              << "  --direction:   force direction (x, y, or z)\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try {
        // Parse CLI arguments
        if (argc < 5) {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 1;
        }

        std::string partition_dir = argv[1];
        std::string config_path   = argv[2];
        std::string output_dir    = argv[3];
        std::string direction;

        for (int i = 4; i < argc; ++i) {
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
            std::cout << "gf_solver: partition=" << partition_dir
                      << " config=" << config_path
                      << " output=" << output_dir
                      << " direction=" << direction << std::endl;
        }

        // Run the forward solver
        int result = gf::run_forward(partition_dir, config_path, output_dir, direction);

        MPI_Finalize();
        return result;

    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Fatal error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
}