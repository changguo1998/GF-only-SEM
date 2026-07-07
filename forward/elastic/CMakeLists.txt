# forward/elastic/CMakeLists.txt — elastic SEM solver executables
#
# Produces:
#   libgf                 (STATIC, backward-compatible — shared + exchange + elastic element + solver)
#   libgf_cuda_nompi      (STATIC, backward-compatible — shared + noop-exchange + elastic element + CUDA + solver)
#   libgf_cuda            (STATIC, backward-compatible — shared + exchange + elastic element + CUDA + solver)
#   gf_solver_mpi         (executable, CPU + MPI)
#   gf_solver_cuda        (executable, CUDA single-GPU, no MPI)
#   gf_solver_mpi_cuda    (executable, CUDA + MPI multi-GPU)

# -----------------------------------------------------------------------
# Elastic element sources (physics-specific)
# -----------------------------------------------------------------------
set(ELASTIC_CPU_SRCS src/element_cpu.cpp)
set(ELASTIC_CUDA_SRCS src/element_cuda.cu)

# -----------------------------------------------------------------------
# Include directories for this subdirectory
# -----------------------------------------------------------------------
set(ELASTIC_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/../share/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../compress/include
    ${HDF5_INCLUDE_DIRS})

# Eigen3 include: compute from Eigen3_DIR
get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
list(APPEND ELASTIC_INCLUDES "${_eigen_prefix}/include/eigen3")

# -----------------------------------------------------------------------
# 1. gf_solver_mpi — CPU backend + MPI
# -----------------------------------------------------------------------
if(MPI_FOUND)
    add_library(libgf STATIC
        ${ELASTIC_CPU_SRCS}
        ../share/src/solver.cpp
        ../share/src/exchange.cpp)
    target_include_directories(libgf PUBLIC ${ELASTIC_INCLUDES})
    target_link_libraries(libgf PUBLIC libgf_shared MPI::MPI_CXX)
    target_compile_definitions(libgf PRIVATE GF_ACTIVE_BACKEND=0)
    target_compile_features(libgf PUBLIC cxx_std_17)

    add_executable(gf_solver_mpi src/main.cpp)
    target_link_libraries(gf_solver_mpi PRIVATE libgf)
    set_target_properties(gf_solver_mpi PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                   "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_mpi (CPU + MPI)")
endif()

# -----------------------------------------------------------------------
# 2. gf_solver_cuda — CUDA backend, no MPI (single GPU)
# -----------------------------------------------------------------------
if(CUDAToolkit_FOUND)
    add_library(libgf_cuda_nompi STATIC
        ${ELASTIC_CPU_SRCS}
        ${ELASTIC_CUDA_SRCS}
        ../share/src/solver.cpp
        ../share/src/exchange_noop.cpp
        ../share/src/cuda_step.cu)
    target_include_directories(libgf_cuda_nompi PUBLIC ${ELASTIC_INCLUDES})
    # Eigen3 include for CUDA: nvcc uses response files — use compile-option -I
    get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
    target_compile_options(libgf_cuda_nompi
                           PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${_eigen_prefix}/include/eigen3>)
    target_link_libraries(libgf_cuda_nompi PUBLIC libgf_shared CUDA::cudart)
    target_compile_definitions(libgf_cuda_nompi PRIVATE GF_ACTIVE_BACKEND=1 GF_WITH_CUDA GF_NO_MPI)
    target_compile_features(libgf_cuda_nompi PUBLIC cxx_std_17)
    target_compile_options(libgf_cuda_nompi
                           PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(libgf_cuda_nompi PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")

    add_executable(gf_solver_cuda src/main.cpp)
    target_link_libraries(gf_solver_cuda PRIVATE libgf_cuda_nompi)
    target_compile_definitions(gf_solver_cuda PRIVATE GF_WITH_CUDA GF_NO_MPI)
    set_target_properties(gf_solver_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_cuda (CUDA, no MPI)")
endif()

# -----------------------------------------------------------------------
# 3. gf_solver_mpi_cuda — CUDA backend + MPI (multi-GPU cluster)
# -----------------------------------------------------------------------
if(CUDAToolkit_FOUND AND MPI_FOUND)
    add_library(libgf_cuda STATIC
        ${ELASTIC_CPU_SRCS}
        ${ELASTIC_CUDA_SRCS}
        ../share/src/solver.cpp
        ../share/src/exchange.cpp
        ../share/src/cuda_step.cu)
    target_include_directories(libgf_cuda PUBLIC ${ELASTIC_INCLUDES})
    # Eigen3 include for CUDA
    get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
    target_compile_options(libgf_cuda
                           PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${_eigen_prefix}/include/eigen3>)
    target_link_libraries(libgf_cuda PUBLIC libgf_shared MPI::MPI_CXX CUDA::cudart)
    target_compile_definitions(libgf_cuda PRIVATE GF_ACTIVE_BACKEND=1 GF_WITH_CUDA)
    target_compile_features(libgf_cuda PUBLIC cxx_std_17)
    target_compile_options(libgf_cuda
                           PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(libgf_cuda PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")

    add_executable(gf_solver_mpi_cuda src/main.cpp)
    target_link_libraries(gf_solver_mpi_cuda PRIVATE libgf_cuda)
    target_compile_definitions(gf_solver_mpi_cuda PRIVATE GF_WITH_CUDA)
    set_target_properties(gf_solver_mpi_cuda PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                         "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_mpi_cuda (CUDA + MPI)")
endif()