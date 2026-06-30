# tools/cpp/CMakeLists.txt — C++ VTK conversion tools
# Built with OpenMP multithreading. Requires HDF5 C library.
# Keep Python editions in tools/*.py — these are accelerated alternatives.

cmake_minimum_required(VERSION 3.16)
project(gf_vtk_tools LANGUAGES CXX)

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP QUIET)

# Common include
set(TOOLS_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# ── model2vtk ──────
add_executable(gf_model2vtk model2vtk.cpp)
target_include_directories(gf_model2vtk PRIVATE ${TOOLS_CPP_DIR} ${HDF5_INCLUDE_DIRS})
target_link_libraries(gf_model2vtk PRIVATE ${HDF5_C_LIBRARIES})

# ── partition2vtk ──
add_executable(gf_partition2vtk partition2vtk.cpp)
target_include_directories(gf_partition2vtk PRIVATE ${TOOLS_CPP_DIR} ${HDF5_INCLUDE_DIRS})
target_link_libraries(gf_partition2vtk PRIVATE ${HDF5_C_LIBRARIES})

# ── wavefield2vtk ──
add_executable(gf_wavefield2vtk wavefield2vtk.cpp)
target_include_directories(gf_wavefield2vtk PRIVATE ${TOOLS_CPP_DIR} ${HDF5_INCLUDE_DIRS})
target_link_libraries(gf_wavefield2vtk PRIVATE ${HDF5_C_LIBRARIES})

# ── OpenMP ──
if(OpenMP_CXX_FOUND)
    foreach(tgt gf_model2vtk gf_partition2vtk gf_wavefield2vtk)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_options(${tgt} PRIVATE ${OpenMP_CXX_FLAGS})
    endforeach()
    message(STATUS "OpenMP found: parallel VTK tools enabled")
else()
    message(STATUS "OpenMP not found: single-threaded VTK tools")
endif()

foreach(tgt gf_model2vtk gf_partition2vtk gf_wavefield2vtk)
    target_compile_features(${tgt} PRIVATE cxx_std_17)
    set_target_properties(${tgt} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                         "${CMAKE_SOURCE_DIR}/bin")
endforeach()