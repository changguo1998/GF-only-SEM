# preprocess/cpp/CMakeLists.txt — C++ preprocessor (unified: stage1 + stage2)
#
# Single executable: gf_preprocess gf_preprocess stage1 <model.h5> [--N N] [--cfl-safety val] ...
# gf_preprocess stage2 <model.h5>

cmake_minimum_required(VERSION 3.16)
project(gf_preprocess_cpp LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP QUIET)
include(GNUInstallDirs)

add_executable(
    gf_preprocess
    gf_preprocess.cpp # unified entry point (stage1/stage2 dispatch)
    main.cpp # stage1: GLL geometry, PML, CFL
    stage2_main.cpp # stage2: λ/μ, solver_dt, pre-flight stats
)

target_include_directories(gf_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR}
                                                 ${HDF5_INCLUDE_DIRS})
target_link_libraries(gf_preprocess PRIVATE Eigen3::Eigen ${HDF5_C_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(gf_preprocess PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(gf_preprocess PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP found: enabled parallel GLL geometry")
else()
    message(STATUS "OpenMP not found: single-threaded GLL geometry")
endif()

target_compile_features(gf_preprocess PRIVATE cxx_std_17)

set_target_properties(gf_preprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

install(TARGETS gf_preprocess RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT gf_tools)
