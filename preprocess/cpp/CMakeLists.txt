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
    gf_preprocess.cpp # unified entry point (stage1/stage2/run dispatch)
    main.cpp # stage1: GLL geometry, PML, CFL
    stage2_main.cpp # stage2: λ/μ, solver_dt, pre-flight stats
)

# ── User-provided material model (optional) ──────────────────────────
set(GF_MATERIAL_USER_SOURCE
    ""
    CACHE
        FILEPATH
        "User-supplied .cpp file implementing gf::evaluate_vp/vs/density.\
     When provided, gf_preprocess calls these directly between stage1\
     and stage2 — no Python or HDF5 round-trip needed.\
     Example: -DGF_MATERIAL_USER_SOURCE=my_material.cpp")

if(GF_MATERIAL_USER_SOURCE)
    if(NOT EXISTS "${GF_MATERIAL_USER_SOURCE}")
        message(FATAL_ERROR "GF_MATERIAL_USER_SOURCE not found: ${GF_MATERIAL_USER_SOURCE}")
    endif()
    target_sources(gf_preprocess PRIVATE "${GF_MATERIAL_USER_SOURCE}")
    target_compile_definitions(gf_preprocess PRIVATE GF_HAS_USER_MATERIAL)
    message(STATUS "User material model: ${GF_MATERIAL_USER_SOURCE}")
else()
    message(STATUS "No user material model — Python/HDF5 path for vp/vs/density")
endif()

target_include_directories(
    gf_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/preprocess/include
                          ${EIGEN3_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
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
