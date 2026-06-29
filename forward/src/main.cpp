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

#ifdef GF_WITH_CUDA
#include <cuda_runtime.h>
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

#ifdef GF_WITH_CUDA
    int n_devices = 0;
    cudaError_t cerr = cudaGetDeviceCount(&n_devices);
    if (cerr != cudaSuccess)
        n_devices = 0;

    if (n_devices == 0) {
        if (rank == 0)
            std::cerr << "Error: no CUDA-capable GPU found.\n";
#ifndef GF_NO_MPI
        MPI_Finalize();
#endif
        return 1;
    }

    cudaSetDevice(rank % n_devices);

#ifndef GF_NO_MPI
    // Per-node warning: MPI ranks on this node > GPUs
    {
        MPI_Comm shm_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
        int shm_size = 0, shm_rank = 0;
        MPI_Comm_size(shm_comm, &shm_size);
        MPI_Comm_rank(shm_comm, &shm_rank);
        MPI_Comm_free(&shm_comm);

        if (shm_rank == 0 && shm_size > n_devices) {
            std::cout << "[WARN] " << shm_size << " MPI ranks on this node but only " << n_devices
                      << " GPU(s) — ranks share GPU(s), performance degraded.\n";
        }
    }
#endif
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