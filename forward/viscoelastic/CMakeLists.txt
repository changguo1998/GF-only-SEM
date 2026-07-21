# forward/viscoelastic/CMakeLists.txt — viscoelastic SEM solver executables (SLS attenuation)
#
# Produces:
#   gf_solver_viscoelastic_mpi       (CPU + MPI)
#   gf_solver_viscoelastic_cuda      (CUDA single-GPU, no MPI)
#   gf_solver_viscoelastic_mpi_cuda  (CUDA + MPI multi-GPU)
#
# Reuses libgf_shared for solver infrastructure.  Only the element kernel
# differs from the elastic solver.

# -----------------------------------------------------------------------
# Viscoelastic element sources (physics-specific)
# -----------------------------------------------------------------------
set(VISCO_CPU_SRCS  src/element_cpu.cpp)
set(VISCO_CUDA_SRCS )  # added in Task 8

# -----------------------------------------------------------------------
# Include directories for this subdirectory
# -----------------------------------------------------------------------
set(VISCO_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../share/include
                   ${CMAKE_CURRENT_SOURCE_DIR}/include
                   ${HDF5_INCLUDE_DIRS})

# Eigen3 include: compute from Eigen3_DIR
get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
list(APPEND VISCO_INCLUDES "${_eigen_prefix}/include/eigen3")

# -----------------------------------------------------------------------
# 1. gf_solver_viscoelastic_mpi — CPU backend + MPI
# -----------------------------------------------------------------------
if(MPI_FOUND)
    add_library(libgf_visco STATIC ${VISCO_CPU_SRCS} src/solver.cpp
                                   ../share/src/exchange.cpp)
    target_include_directories(libgf_visco PUBLIC ${VISCO_INCLUDES})
    target_link_libraries(libgf_visco PUBLIC libgf_shared MPI::MPI_CXX)
    target_compile_definitions(libgf_visco PRIVATE GF_ACTIVE_BACKEND=0)
    target_compile_features(libgf_visco PUBLIC cxx_std_17)

    add_executable(gf_solver_viscoelastic_mpi src/main.cpp)
    target_link_libraries(gf_solver_viscoelastic_mpi PRIVATE libgf_visco)
    set_target_properties(gf_solver_viscoelastic_mpi PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_viscoelastic_mpi (CPU + MPI)")
endif()

# -----------------------------------------------------------------------
# 2. gf_solver_viscoelastic_cuda — CUDA backend, no MPI (single GPU)
# -----------------------------------------------------------------------
if(CUDAToolkit_FOUND)
    add_library(
        libgf_visco_cuda_nompi
        STATIC ${VISCO_CPU_SRCS} ${VISCO_CUDA_SRCS} src/solver.cpp
               ../share/src/exchange_noop.cpp ../share/src/cuda_step.cu)
    target_include_directories(libgf_visco_cuda_nompi PUBLIC ${VISCO_INCLUDES})
    # Eigen3 include for CUDA: nvcc uses response files — use compile-option -I
    get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
    target_compile_options(
        libgf_visco_cuda_nompi
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${_eigen_prefix}/include/eigen3>)
    target_link_libraries(libgf_visco_cuda_nompi PUBLIC libgf_shared CUDA::cudart)
    target_compile_definitions(libgf_visco_cuda_nompi
                               PRIVATE GF_ACTIVE_BACKEND=1 GF_WITH_CUDA GF_NO_MPI)
    target_compile_features(libgf_visco_cuda_nompi PUBLIC cxx_std_17)
    target_compile_options(
        libgf_visco_cuda_nompi
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(libgf_visco_cuda_nompi PROPERTIES
                          CUDA_ARCHITECTURES "80;86;87;90")

    add_executable(gf_solver_viscoelastic_cuda src/main.cpp)
    target_link_libraries(gf_solver_viscoelastic_cuda PRIVATE libgf_visco_cuda_nompi)
    target_compile_definitions(gf_solver_viscoelastic_cuda PRIVATE GF_WITH_CUDA GF_NO_MPI)
    set_target_properties(gf_solver_viscoelastic_cuda PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_viscoelastic_cuda (CUDA, no MPI)")
endif()

# -----------------------------------------------------------------------
# 3. gf_solver_viscoelastic_mpi_cuda — CUDA backend + MPI (multi-GPU)
# -----------------------------------------------------------------------
if(CUDAToolkit_FOUND AND MPI_FOUND)
    add_library(libgf_visco_cuda STATIC ${VISCO_CPU_SRCS} ${VISCO_CUDA_SRCS} src/solver.cpp
                                        ../share/src/exchange.cpp ../share/src/cuda_step.cu)
    target_include_directories(libgf_visco_cuda PUBLIC ${VISCO_INCLUDES})
    # Eigen3 include for CUDA
    get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
    target_compile_options(
        libgf_visco_cuda
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${_eigen_prefix}/include/eigen3>)
    target_link_libraries(libgf_visco_cuda PUBLIC libgf_shared MPI::MPI_CXX CUDA::cudart)
    target_compile_definitions(libgf_visco_cuda PRIVATE GF_ACTIVE_BACKEND=1 GF_WITH_CUDA)
    target_compile_features(libgf_visco_cuda PUBLIC cxx_std_17)
    target_compile_options(
        libgf_visco_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(libgf_visco_cuda PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")

    add_executable(gf_solver_viscoelastic_mpi_cuda src/main.cpp)
    target_link_libraries(gf_solver_viscoelastic_mpi_cuda PRIVATE libgf_visco_cuda)
    target_compile_definitions(gf_solver_viscoelastic_mpi_cuda PRIVATE GF_WITH_CUDA)
    set_target_properties(gf_solver_viscoelastic_mpi_cuda PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

    message(STATUS "  -> gf_solver_viscoelastic_mpi_cuda (CUDA + MPI)")
endif()