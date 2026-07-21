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
                                   ../share/src/solver.cpp
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
# 2-3. CUDA targets — disabled until Task 8 implements the viscoelastic
# CUDA element kernel (viscoelastic/src/element_cuda.cu) and the
# cuda_launch_element_residual() function.
# -----------------------------------------------------------------------
# if(CUDAToolkit_FOUND)
#   ... (enable in Task 8)
# endif()
# if(CUDAToolkit_FOUND AND MPI_FOUND)
#   ... (enable in Task 8)
# endif()