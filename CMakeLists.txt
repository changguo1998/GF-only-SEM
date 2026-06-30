cmake_minimum_required(VERSION 3.18)
project(gf_calculation LANGUAGES CXX)

# -----------------------------------------------------------------------
# Dependencies (auto-detected where available)
# -----------------------------------------------------------------------

# MPI: optional at top level. Required for MPI-based solvers. Without MPI, only gf_solver_cuda
# (single-GPU) can be built.
find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI found — MPI solver targets enabled")
else()
    message(WARNING "MPI not found — MPI-based targets will NOT be built")
endif()

# CUDA: optional. If found, CUDA-accelerated solver targets are built. Must be checked at top level
# because enable_language(CUDA) needs to happen before any CUDA targets are defined in
# subdirectories.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA toolkit found — CUDA solver targets enabled")
else()
    message(STATUS "CUDA toolkit not found — only CPU solver targets will be built")
endif()

# Required dependencies
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Eigen3 REQUIRED)
# Propagate Eigen3 include path to subdirectories (same logic as forward/CMakeLists.txt)
get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
set(EIGEN3_INCLUDE_DIR "${_eigen_prefix}/include/eigen3")

# -----------------------------------------------------------------------
# Enable testing
# -----------------------------------------------------------------------
include(CTest)

# -----------------------------------------------------------------------
# Fetch Catch2
# -----------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4)
FetchContent_MakeAvailable(Catch2)

# -----------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------
add_subdirectory(compress)
add_subdirectory(forward)
add_subdirectory(preprocess/cpp)
add_subdirectory(tests)

# -----------------------------------------------------------------------
# Alias target for convenience
# -----------------------------------------------------------------------
add_custom_target(
    test_all
    COMMAND ctest --output-on-failure -V
    DEPENDS test_compress
            test_gll
            test_element
            test_newmark
            test_pml
            test_source
            test_io
            test_record)

# CUDA element test (if CUDA available)
if(TARGET test_element_cuda)
    add_dependencies(test_all test_element_cuda)
endif()
