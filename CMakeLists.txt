cmake_minimum_required(VERSION 3.16)
project(gf_calculation LANGUAGES CXX)

# ----- Enable testing -----
include(CTest)

# ----- Fetch Catch2 -----
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(Catch2)

add_subdirectory(compress)
add_subdirectory(forward)

# ----- Find packages for tests (and general use) -----
find_package(MPI REQUIRED CXX)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Eigen3 REQUIRED)

add_subdirectory(tests)

# Alias target for convenience
add_custom_target(test_all
    COMMAND ctest --output-on-failure -V
    DEPENDS test_compress test_gll test_element test_newmark test_pml test_source test_io test_record
)