# postprocess/cpp/CMakeLists.txt — C++ accelerated Green's function assembly & tiling
#
# Reads per-step record files from three force-direction runs, merges by vertex_id, assembles
# Green's tensor, and writes spatially-tiled HDF5 output (byte-identical to Python gf_post).
#
# Two targets: gf_postprocess        — serial (original, no MPI) gf_postprocess_mpi    — MPI: one
# rank per tile

cmake_minimum_required(VERSION 3.16)
project(gf_postprocess_cpp LANGUAGES CXX)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(MPI REQUIRED)
include(GNUInstallDirs)

# ---- Serial target ----
add_executable(gf_postprocess main.cpp)
target_include_directories(gf_postprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})
target_link_libraries(gf_postprocess PRIVATE ${HDF5_C_LIBRARIES} ${CMAKE_DL_LIBS})
target_compile_features(gf_postprocess PRIVATE cxx_std_17)
set_target_properties(gf_postprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
install(TARGETS gf_postprocess RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT gf_tools)

# ---- MPI target ----
add_executable(gf_postprocess_mpi main_mpi.cpp)
target_include_directories(gf_postprocess_mpi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                      ${HDF5_INCLUDE_DIRS})
target_link_libraries(gf_postprocess_mpi PRIVATE ${HDF5_C_LIBRARIES} ${CMAKE_DL_LIBS} MPI::MPI_CXX)
target_compile_definitions(gf_postprocess_mpi PRIVATE GF_POST_MPI)
target_compile_features(gf_postprocess_mpi PRIVATE cxx_std_17)
set_target_properties(gf_postprocess_mpi PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${CMAKE_SOURCE_DIR}/bin")
install(TARGETS gf_postprocess_mpi RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT gf_tools)
