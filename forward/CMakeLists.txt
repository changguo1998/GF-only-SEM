# forward/CMakeLists.txt — top-level for the forward solver module
cmake_minimum_required(VERSION 3.18)
project(gf_forward LANGUAGES CXX C)
if(CMAKE_CUDA_COMPILER OR DEFINED ENV{CUDACXX})
    enable_language(CUDA)
endif()
include(GNUInstallDirs)

# -----------------------------------------------------------------------
# Dependencies (propagated to subdirectories)
# -----------------------------------------------------------------------

# MPI: required for MPI-based solvers. Without MPI, only gf_solver_elastic_cuda
find_package(MPI)
if(MPI_FOUND)
    message(STATUS "MPI found — MPI solver targets enabled")
else()
    message(WARNING "MPI not found — MPI solver targets will NOT be built")
endif()

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
# CUDA: optional, for GPU-accelerated solvers
find_package(CUDAToolkit)

# -----------------------------------------------------------------------
# Common solver-target generator (shared by elastic, viscoelastic, ...)
# -----------------------------------------------------------------------

function(add_gf_physics_solvers)
    set(options "")
    set(oneValueArgs SOLVER LIB)
    set(multiValueArgs CPU_ELEM_SRCS CUDA_ELEM_SRCS EXTRA_SRCS EXTRA_INCLUDES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Library / executable naming
    set(LIB_CPU ${ARG_LIB})
    set(LIB_CUDA_NOSPI ${ARG_LIB}_cuda_nompi)
    set(LIB_CUDA ${ARG_LIB}_cuda)
    set(EXE_MPI gf_solver_${ARG_SOLVER}_mpi)
    set(EXE_CUDA gf_solver_${ARG_SOLVER}_cuda)
    set(EXE_MPI_CUDA gf_solver_${ARG_SOLVER}_mpi_cuda)

    # Shared infrastructure (same for every physics variant)
    set(SHARED_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/../share/src/solver.cpp")
    set(MPI_EXCHANGE "${CMAKE_CURRENT_SOURCE_DIR}/../share/src/exchange.cpp")
    set(NOOP_EXCHANGE "${CMAKE_CURRENT_SOURCE_DIR}/../share/src/exchange_noop.cpp")
    set(CUDA_STEP "${CMAKE_CURRENT_SOURCE_DIR}/../share/src/cuda_step.cu")

    # Include paths
    set(INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/../share/include" ${ARG_EXTRA_INCLUDES}
                 "${HDF5_INCLUDE_DIRS}")
    get_filename_component(_ep "${Eigen3_DIR}/../../.." ABSOLUTE)
    list(APPEND INCLUDES "${_ep}/include/eigen3")

    # ── 1. CPU + MPI ──
    if(MPI_FOUND)
        add_library(${LIB_CPU} STATIC ${ARG_CPU_ELEM_SRCS} ${SHARED_SRCS} ${MPI_EXCHANGE}
                                      ${ARG_EXTRA_SRCS})
        target_include_directories(${LIB_CPU} PUBLIC ${INCLUDES})
        target_link_libraries(${LIB_CPU} PUBLIC libgf_shared MPI::MPI_CXX)
        target_compile_definitions(${LIB_CPU} PRIVATE GF_ACTIVE_BACKEND=0)
        target_compile_features(${LIB_CPU} PUBLIC cxx_std_17)

        add_executable(${EXE_MPI} src/main.cpp)
        target_link_libraries(${EXE_MPI} PRIVATE ${LIB_CPU})
        set_target_properties(${EXE_MPI} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${CMAKE_SOURCE_DIR}/bin")
        message(STATUS "  -> ${EXE_MPI} (CPU + MPI)")
        install(TARGETS ${EXE_MPI} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT gf_solver)
    endif()

    # ── 2. CUDA, no MPI (single GPU) ──
    if(CUDAToolkit_FOUND)
        add_library(${LIB_CUDA_NOSPI} STATIC ${ARG_CUDA_ELEM_SRCS} ${SHARED_SRCS} ${NOOP_EXCHANGE}
                                             ${CUDA_STEP} ${ARG_EXTRA_SRCS})
        target_include_directories(${LIB_CUDA_NOSPI} PUBLIC ${INCLUDES})
        target_compile_options(${LIB_CUDA_NOSPI}
                               PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${_ep}/include/eigen3>)
        target_link_libraries(${LIB_CUDA_NOSPI} PUBLIC libgf_shared CUDA::cudart)
        target_compile_definitions(${LIB_CUDA_NOSPI} PRIVATE GF_ACTIVE_BACKEND=1 GF_WITH_CUDA
                                                             GF_NO_MPI)
        target_compile_features(${LIB_CUDA_NOSPI} PUBLIC cxx_std_17)
        target_compile_options(${LIB_CUDA_NOSPI}
                               PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        set_target_properties(${LIB_CUDA_NOSPI} PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")

        add_executable(${EXE_CUDA} src/main.cpp)
        target_link_libraries(${EXE_CUDA} PRIVATE ${LIB_CUDA_NOSPI})
        target_compile_definitions(${EXE_CUDA} PRIVATE GF_WITH_CUDA GF_NO_MPI)
        set_target_properties(${EXE_CUDA} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                     "${CMAKE_SOURCE_DIR}/bin")
        message(STATUS "  -> ${EXE_CUDA} (CUDA, no MPI)")
        install(TARGETS ${EXE_CUDA} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                                            COMPONENT gf_solver)
    endif()

    # ── 3. CUDA + MPI (multi-GPU) ──
    if(CUDAToolkit_FOUND AND MPI_FOUND)
        add_library(${LIB_CUDA} STATIC ${ARG_CUDA_ELEM_SRCS} ${SHARED_SRCS} ${MPI_EXCHANGE}
                                       ${CUDA_STEP} ${ARG_EXTRA_SRCS})
        target_include_directories(${LIB_CUDA} PUBLIC ${INCLUDES})
        target_compile_options(${LIB_CUDA}
                               PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${_ep}/include/eigen3>)
        target_link_libraries(${LIB_CUDA} PUBLIC libgf_shared MPI::MPI_CXX CUDA::cudart)
        target_compile_definitions(${LIB_CUDA} PRIVATE GF_ACTIVE_BACKEND=1 GF_WITH_CUDA)
        target_compile_features(${LIB_CUDA} PUBLIC cxx_std_17)
        target_compile_options(${LIB_CUDA}
                               PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        set_target_properties(${LIB_CUDA} PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")

        add_executable(${EXE_MPI_CUDA} src/main.cpp)
        target_link_libraries(${EXE_MPI_CUDA} PRIVATE ${LIB_CUDA})
        target_compile_definitions(${EXE_MPI_CUDA} PRIVATE GF_WITH_CUDA)
        set_target_properties(${EXE_MPI_CUDA} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                         "${CMAKE_SOURCE_DIR}/bin")
        message(STATUS "  -> ${EXE_MPI_CUDA} (CUDA + MPI)")
        install(TARGETS ${EXE_MPI_CUDA} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                                                COMPONENT gf_solver)
    endif()
endfunction()

# -----------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------
add_subdirectory(share)
add_subdirectory(elastic)
add_subdirectory(viscoelastic)
