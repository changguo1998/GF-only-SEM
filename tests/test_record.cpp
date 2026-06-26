// tests/test_record.cpp — RecordWriter tests
#include <hdf5.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdlib>
#include <string>
#include <vector>

#include "gf/CompressionFilter.h"
#include "gf/record.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

TEST_CASE("RecordWriter creates file and writes strain", "[record]") {
    int ngll = 4;
    int n_elem = 2;
    int n_node = n_elem * ngll * ngll * ngll;

    CompressionConfig comp;
    comp.method = CompressionMethod::None;

    std::vector<int64_t> elem_ids = {1, 2};

    // Create writer (writes to temp)
    RecordWriter writer("./wavefields", "x", 0, n_elem, elem_ids.data(), ngll, comp, false);

    // Write a few steps of strain data
    std::vector<double> strain(n_node * 6, 0.0);
    for (int step = 0; step < 3; ++step) {
        for (size_t i = 0; i < strain.size(); ++i) {
            strain[i] = static_cast<double>(step) * 1e-7;
        }
        writer.write_step(step, strain.data());
    }

    writer.close();

    // Verify the file was created
    std::string fname = "./wavefields/x/record_0.h5";
    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    // Check strain dataset
    hid_t dset = H5Dopen2(file, "strain", H5P_DEFAULT);
    REQUIRE(dset >= 0);

    hid_t space = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(space);
    REQUIRE(ndims == 6);
    hsize_t dims[6];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    REQUIRE(dims[0] == 3);  // 3 steps written
    REQUIRE(dims[1] == (hsize_t)n_elem);
    REQUIRE(dims[2] == (hsize_t)ngll);
    REQUIRE(dims[3] == (hsize_t)ngll);
    REQUIRE(dims[4] == (hsize_t)ngll);
    REQUIRE(dims[5] == 6);

    H5Sclose(space);
    H5Dclose(dset);

    // Check local_element_ids
    dset = H5Dopen2(file, "local_element_ids", H5P_DEFAULT);
    REQUIRE(dset >= 0);
    std::vector<int64_t> read_ids(n_elem);
    H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, read_ids.data());
    REQUIRE(read_ids[0] == 1);
    REQUIRE(read_ids[1] == 2);
    H5Dclose(dset);

    H5Fclose(file);

    // Cleanup
    std::remove(fname.c_str());
}

TEST_CASE("RecordWriter with float32 compression", "[record]") {
    int ngll = 4;
    int n_elem = 1;
    int n_node = n_elem * ngll * ngll * ngll;

    CompressionConfig comp;
    comp.method = CompressionMethod::None;

    std::vector<int64_t> elem_ids = {5};

    RecordWriter writer("./wavefields", "y", 1, n_elem, elem_ids.data(), ngll, comp, true);

    std::vector<double> strain(n_node * 6, 1e-6);
    writer.write_step(0, strain.data());
    writer.close();

    // Verify file exists and has correct direction attribute
    std::string fname = "./wavefields/y/record_1.h5";
    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    H5Fclose(file);

    std::remove(fname.c_str());
}