# tests/CMakeLists.txt — Catch2 tests for forward solver + compress module

if(POLICY CMP0110)
    cmake_policy(SET CMP0110 NEW)
endif()

# Include Catch2 helper module (provides catch_discover_tests)
list(PREPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# ---------------------------------------------------------------
# Forward solver tests (require MPI — link libgf)
# ---------------------------------------------------------------

if(MPI_FOUND)
    function(add_gf_test name source)
        add_executable(${name} ${source})
        target_include_directories(
            ${name}
            PRIVATE ${CMAKE_SOURCE_DIR}/forward/include ${CMAKE_SOURCE_DIR}/compress/include
                    ${EIGEN3_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE libgf Catch2::Catch2WithMain MPI::MPI_CXX
                                              Eigen3::Eigen ${HDF5_C_LIBRARIES})
        target_compile_features(${name} PRIVATE cxx_std_17)
        catch_discover_tests(${name})
    endfunction()

    # GLL quadrature tests
    add_gf_test(test_gll test_gll.cpp)

    # Element residual tests (core physics — matrix-free K·u)
    add_gf_test(test_element test_element.cpp)

    # Newmark time integration tests
    add_gf_test(test_newmark test_newmark.cpp)

    # PML damping tests
    add_gf_test(test_pml test_pml.cpp)

    # Source injection tests
    add_gf_test(test_source test_source.cpp)

    # I/O round-trip tests (HDF5)
    add_gf_test(test_io test_io.cpp)

    # Record writer tests (HDF5)
    add_gf_test(test_record test_record.cpp)

    # Assembly tests
    add_gf_test(test_assembly test_assembly.cpp)

    # Exchange tests (MPI, requires mpirun -np 2 for multi-rank cases)
    add_executable(test_exchange test_exchange.cpp)
    target_include_directories(
        test_exchange
        PRIVATE ${CMAKE_SOURCE_DIR}/forward/include ${CMAKE_SOURCE_DIR}/compress/include
                ${EIGEN3_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
    target_link_libraries(test_exchange PRIVATE libgf Catch2::Catch2 MPI::MPI_CXX Eigen3::Eigen
                                                ${HDF5_C_LIBRARIES})
    target_compile_features(test_exchange PRIVATE cxx_std_17)

    add_test(NAME "Exchange halo transfers data between two ranks"
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_exchange>
                     "Exchange halo transfers data between two ranks")
    add_test(NAME "Exchange accumulate adds received data (CG-SEM assembly)"
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:test_exchange>
                     "Exchange accumulate adds received data (CG-SEM assembly)")
    add_test(NAME "Exchange with empty patterns is no-op"
             COMMAND $<TARGET_FILE:test_exchange> "Exchange with empty patterns is no-op")

    # Integration tests (single-element end-to-end)
    add_gf_test(test_integration test_integration.cpp)

    message(STATUS "MPI-dependent forward tests enabled")
else()
    message(WARNING "MPI not found — forward solver tests disabled")
endif()

# ---------------------------------------------------------------
# CUDA element residual tests (requires CUDA toolkit, no MPI needed)
# ---------------------------------------------------------------
if(TARGET libgf_cuda_nompi)
    add_executable(test_element_cuda test_element_cuda.cu)
    target_include_directories(
        test_element_cuda
        PRIVATE ${CMAKE_SOURCE_DIR}/forward/include ${CMAKE_SOURCE_DIR}/compress/include
                ${EIGEN3_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
    # Eigen3 include for CUDA (nvcc response files don't see CPATH).
    # Use COMPILE_LANGUAGE generator expression to ensure nvcc picks it up.
    get_filename_component(_eigen_prefix "${Eigen3_DIR}/../../.." ABSOLUTE)
    target_include_directories(
        test_element_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${_eigen_prefix}/include/eigen3>)
    target_link_libraries(test_element_cuda PRIVATE libgf_cuda_nompi Catch2::Catch2WithMain
                                                    Eigen3::Eigen ${HDF5_C_LIBRARIES} CUDA::cudart)
    target_compile_features(test_element_cuda PRIVATE cxx_std_17)
    target_compile_definitions(test_element_cuda PRIVATE GF_WITH_CUDA GF_ACTIVE_BACKEND=1)
    target_compile_options(test_element_cuda
                           PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(test_element_cuda PROPERTIES CUDA_ARCHITECTURES "80;86;87;90")
    catch_discover_tests(test_element_cuda)

    message(STATUS "CUDA element tests enabled")
endif()

# ---------------------------------------------------------------
# Compress module tests (always available, no MPI needed)
# ---------------------------------------------------------------
add_executable(test_compress test_compress.cpp)
target_link_libraries(test_compress PRIVATE gf_compress Catch2::Catch2WithMain)
target_compile_features(test_compress PRIVATE cxx_std_17)
catch_discover_tests(test_compress)
