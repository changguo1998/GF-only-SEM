cmake_minimum_required(VERSION 3.18)
project(gf_forward LANGUAGES CXX)

# -----------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------

# MPI: required for MPI-based solvers. If not found, only gf_solver_cuda (single-GPU, no MPI) can
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
# Common source files (no MPI-specific calls, no backend-specific code)
# -----------------------------------------------------------------------
set(COMMON_SRCS
    src/assembly.cpp
    src/pml.cpp
    src/newmark.cpp
    src/source.cpp
    src/io.cpp
    src/record.cpp
    src/restart.cpp
    src/solver.cpp)

# -----------------------------------------------------------------------
# Common include directories and link libraries
# -----------------------------------------------------------------------
set(COMMON_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../compress/include ${HDF5_INCLUDE_DIRS})
# Eigen3 include: compute from Eigen3_DIR since EIGEN3_INCLUDE_DIR may not be set by
# Eigen3Config.cmake in all environments.
get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
list(APPEND COMMON_INCLUDES "${_eigen_prefix}/include/eigen3")
set(COMMON_LIBS Eigen3::Eigen ${HDF5_C_LIBRARIES})

# -----------------------------------------------------------------------
# 1. gf_solver_mpi — CPU backend + MPI
# -----------------------------------------------------------------------
# Produces: libgf           (STATIC, used by tests — backward compatible) gf_solver_mpi
# (executable)

if(MPI_FOUND)
    add_library(libgf STATIC ${COMMON_SRCS} src/exchange.cpp src/element_cpu.cpp)
    target_include_directories(libgf PUBLIC ${COMMON_INCLUDES})
    target_link_libraries(libgf PUBLIC ${COMMON_LIBS} MPI::MPI_CXX)
    target_compile_definitions(libgf PRIVATE GF_ACTIVE_BACKEND=0)
    target_compile_features(libgf PUBLIC cxx_std_17)

    add_executable(gf_solver_mpi src/main.cpp)
    target_link_libraries(gf_solver_mpi PRIVATE libgf)
    set_target_properties(gf_solver_mpi PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                   "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_mpi (CPU + MPI)")
endif()

# -----------------------------------------------------------------------
# 1. gf_solver_cuda — CUDA backend, no MPI (single GPU)
# -----------------------------------------------------------------------
# Produces: libgf_cuda_nompi (STATIC) gf_solver_cuda   (executable)

if(CUDAToolkit_FOUND)
    add_library(libgf_cuda_nompi STATIC ${COMMON_SRCS} src/exchange_noop.cpp src/element_cpu.cpp
                                        src/element_cuda.cu)
    target_include_directories(libgf_cuda_nompi PUBLIC ${COMMON_INCLUDES})
    # Eigen3 include for CUDA: nvcc uses response files and doesn't see CPATH, so
    # target_link_libraries propagation may not reach CUDA.
    get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
    target_include_directories(libgf_cuda_nompi PRIVATE "${_eigen_prefix}/include/eigen3")
    target_link_libraries(libgf_cuda_nompi PUBLIC ${COMMON_LIBS} CUDA::cudart)
    target_compile_definitions(libgf_cuda_nompi PRIVATE GF_ACTIVE_BACKEND=1 GF_WITH_CUDA GF_NO_MPI)
    target_compile_features(libgf_cuda_nompi PUBLIC cxx_std_17)
    target_compile_options(libgf_cuda_nompi
                           PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(libgf_cuda_nompi PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")

    add_executable(gf_solver_cuda src/main.cpp)
    target_link_libraries(gf_solver_cuda PRIVATE libgf_cuda_nompi)
    target_compile_definitions(gf_solver_cuda PRIVATE GF_WITH_CUDA)
    set_target_properties(gf_solver_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_cuda (CUDA, no MPI)")
endif()

# -----------------------------------------------------------------------
# 1. gf_solver_mpi_cuda — CUDA backend + MPI (multi-GPU cluster)
# -----------------------------------------------------------------------
# Produces: libgf_cuda          (STATIC) gf_solver_mpi_cuda  (executable)

if(CUDAToolkit_FOUND AND MPI_FOUND)
    add_library(libgf_cuda STATIC ${COMMON_SRCS} src/exchange.cpp src/element_cpu.cpp
                                  src/element_cuda.cu)
    target_include_directories(libgf_cuda PUBLIC ${COMMON_INCLUDES})
    # Eigen3 include for CUDA: nvcc uses response files and doesn't see CPATH, so
    # target_link_libraries propagation may not reach CUDA.
    get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
    target_include_directories(libgf_cuda PRIVATE "${_eigen_prefix}/include/eigen3")
    target_link_libraries(libgf_cuda PUBLIC ${COMMON_LIBS} MPI::MPI_CXX CUDA::cudart)
    target_compile_definitions(libgf_cuda PRIVATE GF_ACTIVE_BACKEND=1 GF_WITH_CUDA)
    target_compile_features(libgf_cuda PUBLIC cxx_std_17)
    target_compile_options(libgf_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(libgf_cuda PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")

    add_executable(gf_solver_mpi_cuda src/main.cpp)
    target_link_libraries(gf_solver_mpi_cuda PRIVATE libgf_cuda)
    target_compile_definitions(gf_solver_mpi_cuda PRIVATE GF_WITH_CUDA)
    set_target_properties(gf_solver_mpi_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                        "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_mpi_cuda (CUDA + MPI)")
endif()
