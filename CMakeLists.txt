cmake_minimum_required(VERSION 3.18)
project(gf_calculation LANGUAGES CXX)
include(GNUInstallDirs)

# Single build-time switch for diagnostic code and tests.
option(DEBUG "Build diagnostic record, profiling, verbose logging, and tests" OFF)
if(DEBUG)
    add_compile_definitions(DEBUG=1)
    set(GF_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin-debug")
    set(BUILD_TESTING
        ON
        CACHE BOOL "Build tests in DEBUG mode" FORCE)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE
            Debug
            CACHE STRING "CMake build type" FORCE)
    endif()
    message(STATUS "DEBUG build: diagnostics and tests enabled")
else()
    set(GF_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
    set(BUILD_TESTING
        OFF
        CACHE BOOL "Tests are excluded from release builds" FORCE)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE
            Release
            CACHE STRING "CMake build type" FORCE)
    endif()
    message(STATUS "Release build: diagnostic code and tests excluded")
endif()

# -----------------------------------------------------------------------
# Dependencies (auto-detected where available)
# -----------------------------------------------------------------------

# MPI: optional at top level. Required for MPI-based solvers. Without MPI, only
# gf_solver_elastic_cuda (single-GPU) can be built.
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
# Propagate Eigen3 include path to subdirectories (same logic as forward/share/CMakeLists.txt)
get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
set(EIGEN3_INCLUDE_DIR "${_eigen_prefix}/include/eigen3")

# -----------------------------------------------------------------------
# Enable testing only in DEBUG builds
# -----------------------------------------------------------------------
if(DEBUG)
    include(CTest)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.4)
    FetchContent_MakeAvailable(Catch2)
endif()

# -----------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------
add_subdirectory(forward)
add_subdirectory(tools/cpp)
add_subdirectory(postprocess/cpp)
add_subdirectory(preprocess/cpp)

if(DEBUG)
    add_subdirectory(tests)

    # Alias target for convenience
    add_custom_target(
        test_all
        COMMAND ctest --output-on-failure -V
        DEPENDS test_gll
                test_element
                test_newmark
                test_pml
                test_source
                test_io
                test_record
                test_record_release)

    if(TARGET test_element_cuda)
        add_dependencies(test_all test_element_cuda)
    endif()
endif()

# -----------------------------------------------------------------------
# Install scripts (convenience wrappers)
# -----------------------------------------------------------------------
install(
    PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/solver.sh ${CMAKE_SOURCE_DIR}/scripts/build.sh
             ${CMAKE_SOURCE_DIR}/scripts/env.sh
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT gf_scripts)

# -----------------------------------------------------------------------
# CPack (packaging support)
# -----------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "gf-calculation")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "3D viscoelastic SEM forward solver with Green's function extraction")
set(CPACK_PACKAGE_VENDOR "gf-calculation")
set(CPACK_GENERATOR "TGZ;DEB;RPM")
include(CPack)
