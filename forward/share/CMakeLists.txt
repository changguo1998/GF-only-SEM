# forward/share/CMakeLists.txt — shared solver infrastructure (physics-agnostic)
#
# Produces: libgf_shared (STATIC)
#   Core solver components that do NOT depend on MPI/CUDA compile-time flags.
#   Flags-dependent sources (solver.cpp, exchange.cpp, cuda_step.cu) and
#   solver-specific sources (element residual) are compiled per-target
#   in each solver's CMakeLists.txt.

set(SHARED_SRCS
    src/assembly.cpp
    src/newmark.cpp
    src/pml.cpp
    src/source.cpp
    src/io.cpp
    src/record.cpp
    src/restart.cpp)

add_library(libgf_shared STATIC ${SHARED_SRCS})

target_include_directories(libgf_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../compress/include
    ${HDF5_INCLUDE_DIRS})

# Eigen3 include: compute from Eigen3_DIR since EIGEN3_INCLUDE_DIR may not be
# set by Eigen3Config.cmake in all environments.
get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
target_include_directories(libgf_shared PUBLIC "${_eigen_prefix}/include/eigen3")

target_compile_features(libgf_shared PUBLIC cxx_std_17)
target_link_libraries(libgf_shared PUBLIC Eigen3::Eigen ${HDF5_C_LIBRARIES})