# forward/CMakeLists.txt — top-level for the forward solver module
cmake_minimum_required(VERSION 3.18)
project(gf_forward LANGUAGES CXX)

# -----------------------------------------------------------------------
# Dependencies (propagated to subdirectories)
# -----------------------------------------------------------------------

# MPI: required for MPI-based solvers. Without MPI, only gf_solver_cuda (single-GPU, no MPI) can
# be built.
find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI found — MPI solver targets enabled")
else()
    message(WARNING "MPI not found — MPI solver targets will NOT be built")
endif()

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# -----------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------
add_subdirectory(share)
add_subdirectory(elastic)
# add_subdirectory(viscoelastic)   # uncomment when viscoelastic solver is implemented