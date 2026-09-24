# preprocess/cpp/CMakeLists.txt — C++ preprocessor

cmake_minimum_required(VERSION 3.16)
project(gf_preprocess_cpp LANGUAGES CXX)

get_filename_component(GF_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(NOT DEFINED GF_RUNTIME_OUTPUT_DIRECTORY)
    set(GF_RUNTIME_OUTPUT_DIRECTORY "${GF_PROJECT_ROOT}/bin")
endif()

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP QUIET)
include(GNUInstallDirs)
find_path(GF_EIGEN_INCLUDE_DIR Eigen/Dense PATH_SUFFIXES eigen3 REQUIRED)

add_executable(
    gf_preprocess
    gf_preprocess.cpp
    main.cpp
    stage2_main.cpp
    source_locator.cpp
    cpml.cpp
    metis_partition.cpp
    partition_writer.cpp
    config_writer.cpp)

# ── METIS (spack-installed) ───────────────────────────────────────────────── METIS is a required
# spack package for the C++ path. Search the spack install tree and standard system paths.
find_library(
    METIS_LIBRARY metis
    HINTS ${METIS_ROOT}/lib $ENV{METIS_ROOT}/lib
    PATHS /usr/lib /usr/local/lib
    PATH_SUFFIXES lib)
find_path(
    METIS_INCLUDE_DIR metis.h
    HINTS ${METIS_ROOT}/include $ENV{METIS_ROOT}/include
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES include)
if(METIS_LIBRARY AND METIS_INCLUDE_DIR)
    message(STATUS "METIS: ${METIS_LIBRARY}")
else()
    # Try spack location
    execute_process(
        COMMAND spack location -i metis
        OUTPUT_VARIABLE _spack_metis_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_spack_metis_dir)
        find_library(
            METIS_LIBRARY metis
            HINTS ${_spack_metis_dir}/lib
            NO_DEFAULT_PATH)
        find_path(
            METIS_INCLUDE_DIR metis.h
            HINTS ${_spack_metis_dir}/include
            NO_DEFAULT_PATH)
    endif()
endif()
if(NOT METIS_LIBRARY)
    message(FATAL_ERROR "METIS library not found. Install: spack install metis")
endif()

# ── Config source ──────────────────────────────────────────────────── ── Config source (default
# or user-provided) ───────────────────────────────

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
if(GF_USER_CONFIG)
    target_compile_definitions(gf_preprocess PRIVATE GF_HAS_USER_MATERIAL)
endif()

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
    gf_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${GF_PROJECT_ROOT}/preprocess/include
                          ${GF_EIGEN_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS} ${METIS_INCLUDE_DIR})
target_link_libraries(gf_preprocess PRIVATE Eigen3::Eigen ${HDF5_C_LIBRARIES} ${METIS_LIBRARY})
# Some Spack Eigen packages expose an imported target without propagating its include path.
target_compile_options(gf_preprocess PRIVATE "-I${GF_EIGEN_INCLUDE_DIR}")

if(OpenMP_CXX_FOUND)
    target_link_libraries(gf_preprocess PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(gf_preprocess PRIVATE ${OpenMP_CXX_FLAGS})
    message(STATUS "OpenMP: enabled")
else()
    message(STATUS "OpenMP: not found (single-threaded)")
endif()

target_compile_features(gf_preprocess PRIVATE cxx_std_17)
set_target_properties(gf_preprocess PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                               "${GF_RUNTIME_OUTPUT_DIRECTORY}")
install(TARGETS gf_preprocess RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT gf_tools)
