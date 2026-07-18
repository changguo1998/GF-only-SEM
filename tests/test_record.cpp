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

    std::vector<int64_t> vertex_ids = {1, 2};
    std::vector<int32_t> src_elem_local = {0, 0};
    std::vector<int32_t> src_corner = {0, 1};
    RankData::RecordingMap rec_map;
    rec_map.has_recording = true;
    rec_map.vertex_ids = vertex_ids;
    rec_map.src_elem_local = src_elem_local;
    rec_map.src_corner = src_corner;

    // Create writer (writes to temp)
    RecordWriter writer("./wavefields", "x", 0, rec_map, ngll, comp, false);

    // Write a few steps of strain data
    std::vector<double> strain(n_node * 6, 0.0);
    for (int step = 0; step < 3; ++step) {
        for (size_t i = 0; i < strain.size(); ++i) {
            strain[i] = static_cast<double>(step) * 1e-7;
        }
        writer.write_step(step, strain.data());
    }

    writer.close();

    // Verify per-step files (record_{rank}_{step}.h5 format, 1 step per file)
    // Step 0
    {
        std::string fname0 = "./wavefields/x/record_0_0.h5";
        hid_t file0 = H5Fopen(fname0.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file0 >= 0);
        hid_t dset0 = H5Dopen2(file0, "strain", H5P_DEFAULT);
        REQUIRE(dset0 >= 0);
        hid_t space0 = H5Dget_space(dset0);
        REQUIRE(H5Sget_simple_extent_ndims(space0) == 3);
        hsize_t dims0[3];
        H5Sget_simple_extent_dims(space0, dims0, nullptr);
        REQUIRE(dims0[0] == 1);  // per-step: singleton step dim
        REQUIRE(dims0[1] == (hsize_t)n_elem);
        REQUIRE(dims0[2] == 6);  // 6 strain components
        H5Sclose(space0);
        H5Dclose(dset0);
        H5Fclose(file0);
        std::remove(fname0.c_str());
    }
    // Step 1
    {
        std::string fname1 = "./wavefields/x/record_0_1.h5";
        hid_t file1 = H5Fopen(fname1.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file1 >= 0);
        H5Fclose(file1);
        std::remove(fname1.c_str());
    }
    // Step 2
    {
        std::string fname2 = "./wavefields/x/record_0_2.h5";
        hid_t file2 = H5Fopen(fname2.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file2 >= 0);
        H5Fclose(file2);
        std::remove(fname2.c_str());
    }
}

TEST_CASE("RecordWriter with float32 compression", "[record]") {
    int ngll = 4;
    int n_elem = 1;
    int n_node = n_elem * ngll * ngll * ngll;

    CompressionConfig comp;
    comp.method = CompressionMethod::None;

    std::vector<int64_t> vertex_ids2 = {5};
    std::vector<int32_t> src_elem_local2 = {0};
    std::vector<int32_t> src_corner2 = {0};
    RankData::RecordingMap rec_map2;
    rec_map2.has_recording = true;
    rec_map2.vertex_ids = vertex_ids2;
    rec_map2.src_elem_local = src_elem_local2;
    rec_map2.src_corner = src_corner2;

    RecordWriter writer("./wavefields", "y", 1, rec_map2, ngll, comp, true);

    std::vector<double> strain(n_node * 6, 1e-6);
    writer.write_step(0, strain.data());
    writer.close();

    // Verify step file exists (per-step format: record_{rank}_{step}.h5)
    std::string fname = "./wavefields/y/record_1_0.h5";
    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    H5Fclose(file);

    std::remove(fname.c_str());
}