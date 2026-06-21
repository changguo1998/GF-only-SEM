# tests/CMakeLists.txt — Catch2 tests for forward solver + compress module

# Include Catch2 helper module (provides catch_discover_tests)
list(PREPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# ---------------------------------------------------------------
# Forward solver tests (no MPI — test libgf components directly)
# ---------------------------------------------------------------

# Helper: test target that links libgf without MPI for single-rank testing
function(add_gf_test name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/forward/include
        ${CMAKE_SOURCE_DIR}/compress/include
        ${EIGEN3_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIRS}
    )
    target_link_libraries(${name} PRIVATE
        libgf
        Catch2::Catch2WithMain
        MPI::MPI_CXX
        Eigen3::Eigen
        ${HDF5_C_LIBRARIES}
    )
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

# Exchange tests (MPI, requires mpirun -np 2)
add_gf_test(test_exchange test_exchange.cpp)

# Integration tests (single-element end-to-end)
add_gf_test(test_integration test_integration.cpp)

# ---------------------------------------------------------------
# Compress module tests
# ---------------------------------------------------------------
add_executable(test_compress test_compress.cpp)
target_link_libraries(test_compress PRIVATE
    gf_compress
    Catch2::Catch2WithMain
)
target_compile_features(test_compress PRIVATE cxx_std_17)
catch_discover_tests(test_compress)