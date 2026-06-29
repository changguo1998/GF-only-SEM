cmake_minimum_required(VERSION 3.18)
project(gf_calculation LANGUAGES CXX)

# ----- Device backend option (must be set before subdirectories) -----
set(GF_DEVICE_BACKEND
    "CPU"
    CACHE STRING "Device backend: CPU, CUDA")
set_property(CACHE GF_DEVICE_BACKEND PROPERTY STRINGS CPU CUDA)

# ----- Enable testing -----
include(CTest)

# ----- Fetch Catch2 -----
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4)
FetchContent_MakeAvailable(Catch2)

add_subdirectory(compress)
add_subdirectory(forward)
add_subdirectory(preprocess/cpp)

# ----- Find packages for tests (and general use) -----
find_package(MPI REQUIRED CXX)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Eigen3 REQUIRED)

add_subdirectory(tests)

# Alias target for convenience
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

# Add CUDA test target if backend is CUDA
if(GF_DEVICE_BACKEND STREQUAL "CUDA")
    add_dependencies(test_all test_element_cuda)
endif()
