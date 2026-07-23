# forward/elastic/CMakeLists.txt — elastic SEM solver executables
#
# Produces (via add_gf_physics_solvers): libgf_elastic                  STATIC — CPU+MPI elastic
# library libgf_elastic_cuda_nompi       STATIC — CUDA+noop-exchange elastic library
# libgf_elastic_cuda             STATIC — CUDA+MPI elastic library gf_solver_elastic_mpi        CPU
# + MPI gf_solver_elastic_cuda       CUDA single-GPU, no MPI gf_solver_elastic_mpi_cuda   CUDA +
# MPI multi-GPU

# Elastic element sources
set(ELASTIC_CPU_SRCS src/element_cpu.cpp)
set(ELASTIC_CUDA_SRCS src/element_cuda.cu)

add_gf_physics_solvers(
    SOLVER
    elastic
    LIB
    libgf_elastic
    CPU_ELEM_SRCS
    ${ELASTIC_CPU_SRCS}
    CUDA_ELEM_SRCS
    ${ELASTIC_CUDA_SRCS})
