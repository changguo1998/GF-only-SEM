// forward/src/main.cpp
#include <mpi.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "gf_solver initialized" << std::endl;
    }

    MPI_Finalize();
    return 0;
}