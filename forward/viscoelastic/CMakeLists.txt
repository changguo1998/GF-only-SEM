# forward/viscoelastic/CMakeLists.txt — viscoelastic SEM solver executables (SLS attenuation)
#
# Produces (via add_gf_physics_solvers): libgf_visco                  STATIC — CPU+MPI viscoelastic
# library libgf_visco_cuda_nompi       STATIC — CUDA+noop-exchange viscoelastic library
# libgf_visco_cuda             STATIC — CUDA+MPI viscoelastic library gf_solver_viscoelastic_mpi
# CPU + MPI gf_solver_viscoelastic_cuda       CUDA single-GPU, no MPI
# gf_solver_viscoelastic_mpi_cuda   CUDA + MPI multi-GPU
#
# Reuses libgf_shared for solver infrastructure. Only the element kernel and the thin viscoelastic
# solver wrapper (src/solver.cpp, delegates to the shared run_forward()) differ from the elastic
# solver.

# Viscoelastic element sources
set(VISCO_CPU_SRCS src/element_cpu.cpp)
set(VISCO_CUDA_SRCS src/element_cuda.cu)

add_gf_physics_solvers(
    SOLVER
    viscoelastic
    LIB
    libgf_visco
    CPU_ELEM_SRCS
    ${VISCO_CPU_SRCS}
    CUDA_ELEM_SRCS
    ${VISCO_CUDA_SRCS}
    EXTRA_SRCS
    src/solver.cpp
    EXTRA_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
