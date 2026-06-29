cmake_minimum_required(VERSION 3.18)
project(gf_forward LANGUAGES CXX)

# -----------------------------------------------------------------------
# Device backend selection
# -----------------------------------------------------------------------
set(GF_DEVICE_BACKEND
    "CPU"
    CACHE STRING "Device backend: CPU, CUDA")
set_property(CACHE GF_DEVICE_BACKEND PROPERTY STRINGS CPU CUDA)

# Validate and set backend sources + definitions
set(BACKEND_DEFINES "")
set(BACKEND_SRCS "src/element_cpu.cpp")

if(GF_DEVICE_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    list(APPEND BACKEND_SRCS "src/element_cuda.cu")
    list(APPEND BACKEND_DEFINES GF_WITH_CUDA)
    list(APPEND BACKEND_DEFINES GF_ACTIVE_BACKEND=1) # 1 = BackendCUDA
    message(STATUS "CUDA backend enabled")
else()
    list(APPEND BACKEND_DEFINES GF_ACTIVE_BACKEND=0) # 0 = BackendCPU
    message(STATUS "CPU backend enabled (default)")
endif()

# -----------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# -----------------------------------------------------------------------
# Library
# -----------------------------------------------------------------------
add_library(
    libgf STATIC
    src/assembly.cpp
    src/pml.cpp
    src/newmark.cpp
    src/source.cpp
    src/exchange.cpp
    src/io.cpp
    src/record.cpp
    src/restart.cpp
    src/solver.cpp
    ${BACKEND_SRCS})

target_include_directories(
    libgf
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../compress/include
           ${EIGEN3_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})

target_link_libraries(libgf PUBLIC MPI::MPI_CXX Eigen3::Eigen ${HDF5_C_LIBRARIES})

target_compile_definitions(libgf PRIVATE ${BACKEND_DEFINES})
target_compile_features(libgf PUBLIC cxx_std_17)

# CUDA-specific compile options
if(GF_DEVICE_BACKEND STREQUAL "CUDA")
    # CUDA 13.2+ supports sm_75+; adjust for your GPU generation Common: sm_80 (Ampere A100), sm_86
    # (Ampere RTX30xx), sm_90 (Hopper H100)
    set_target_properties(libgf PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")
    # Relax constexpr for Eigen host/device compatibility
    target_compile_options(libgf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# -----------------------------------------------------------------------
# Executable
# -----------------------------------------------------------------------
add_executable(gf_solver src/main.cpp)
target_link_libraries(gf_solver PRIVATE libgf)
set_target_properties(gf_solver PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
