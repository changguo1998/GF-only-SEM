# preprocess/cpp/CMakeLists.txt — C++ preprocessor for GLL geometry, CFL, PML

cmake_minimum_required(VERSION 3.16)
project(gf_preprocess_cpp LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

add_executable(
    gf_preprocess_cpp
    main.cpp)

target_include_directories(
    gf_preprocess_cpp
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
            ${EIGEN3_INCLUDE_DIR}
            ${HDF5_INCLUDE_DIRS})

target_link_libraries(
    gf_preprocess_cpp
    PRIVATE Eigen3::Eigen
            ${HDF5_C_LIBRARIES})

target_compile_features(gf_preprocess_cpp PRIVATE cxx_std_17)