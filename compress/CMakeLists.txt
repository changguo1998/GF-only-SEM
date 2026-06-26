# compress/CMakeLists.txt — Header-only interface library
add_library(gf_compress INTERFACE)
target_include_directories(gf_compress INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# HDF5 dependency (inherited from the forward solver's FindHDF5)
find_package(HDF5 REQUIRED COMPONENTS C)
target_link_libraries(gf_compress INTERFACE HDF5::HDF5)

# C++17 required
target_compile_features(gf_compress INTERFACE cxx_std_17)
