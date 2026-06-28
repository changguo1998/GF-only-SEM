cmake_minimum_required(VERSION 3.16)
project(gf_forward LANGUAGES CXX)

find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

add_library(
    libgf STATIC
    src/element.cpp
    src/assembly.cpp
    src/pml.cpp
    src/newmark.cpp
    src/source.cpp
    src/exchange.cpp
    src/io.cpp
    src/record.cpp
    src/restart.cpp
    src/solver.cpp)

target_include_directories(
    libgf
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../compress/include
           ${EIGEN3_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})

target_link_libraries(libgf PUBLIC MPI::MPI_CXX Eigen3::Eigen ${HDF5_C_LIBRARIES})

add_executable(gf_solver src/main.cpp)
target_link_libraries(gf_solver PRIVATE libgf)

set_target_properties(gf_solver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
