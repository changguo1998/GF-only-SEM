# preprocess/cpp/CMakeLists.txt — C++ preprocessor

cmake_minimum_required(VERSION 3.16)
project(gf_preprocess_cpp LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP QUIET)
include(GNUInstallDirs)

# ── Core sources ────────────────────────────────────────────────────────────

add_executable(gf_preprocess gf_preprocess.cpp main.cpp stage2_main.cpp source_locator.cpp)

# ── Config source (default or user-provided) ───────────────────────────────

set(_config_source "${CMAKE_CURRENT_SOURCE_DIR}/config_default.cpp")
set(GF_USER_CONFIG
    ""
    CACHE FILEPATH "Path to user config.cpp (overrides config_default.cpp)")

if(GF_USER_CONFIG)
    if(NOT EXISTS "${GF_USER_CONFIG}")
        message(FATAL_ERROR "GF_USER_CONFIG not found: ${GF_USER_CONFIG}")
    endif()
    set(_config_source "${GF_USER_CONFIG}")
    message(STATUS "User config: ${GF_USER_CONFIG}")
else()
    message(STATUS "Using default config (config_default.cpp)")
endif()

target_sources(gf_preprocess PRIVATE "${_config_source}")

# ── Legacy material-only source (deprecated) ───────────────────────────────

set(GF_MATERIAL_USER_SOURCE
    ""
    CACHE FILEPATH "Path to material-only .cpp (deprecated, use GF_USER_CONFIG)")

if(GF_MATERIAL_USER_SOURCE AND NOT GF_USER_CONFIG)
    if(NOT EXISTS "${GF_MATERIAL_USER_SOURCE}")
        message(FATAL_ERROR "GF_MATERIAL_USER_SOURCE not found: ${GF_MATERIAL_USER_SOURCE}")
    endif()
    target_sources(gf_preprocess PRIVATE "${GF_MATERIAL_USER_SOURCE}")
    target_compile_definitions(gf_preprocess PRIVATE GF_HAS_USER_MATERIAL)
    message(STATUS "User material (legacy): ${GF_MATERIAL_USER_SOURCE}")
endif()

# ── Include paths, libraries ────────────────────────────────────────────────

target_include_directories(
    gf_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/preprocess/include
                          ${EIGEN3_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
target_link_libraries(gf_preprocess PRIVATE Eigen3::Eigen ${HDF5_C_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(gf_preprocess PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(gf_preprocess PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP: enabled")
else()
    message(STATUS "OpenMP: not found (single-threaded)")
endif()

target_compile_features(gf_preprocess PRIVATE cxx_std_17)
set_target_properties(gf_preprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
install(TARGETS gf_preprocess RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT gf_tools)
