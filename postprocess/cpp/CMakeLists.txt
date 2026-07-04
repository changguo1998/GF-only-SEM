# postprocess/cpp/CMakeLists.txt — C++ accelerated Green's function assembly & tiling
#
# Reads per-step record files from three force-direction runs, merges by vertex_id, assembles
# Green's tensor, and writes spatially-tiled HDF5 output (byte-identical to Python gf_post).
#
# CPU only (no MPI, no CUDA). OpenMP loop for tile write.

cmake_minimum_required(VERSION 3.16)
project(gf_postprocess_cpp LANGUAGES CXX)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP QUIET)

add_executable(gf_postprocess main.cpp)

target_include_directories(gf_postprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})

target_link_libraries(gf_postprocess PRIVATE ${HDF5_C_LIBRARIES} ${CMAKE_DL_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(gf_postprocess PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(gf_postprocess PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP found: parallel tile writing enabled")
else()
    message(STATUS "OpenMP not found: single-threaded tile writing")
endif()

target_compile_features(gf_postprocess PRIVATE cxx_std_17)

set_target_properties(gf_postprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
