// tests/test_record.cpp — RecordWriter tests (GLL-node format)
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

TEST_CASE("RecordWriter creates file and writes GLL strain", "[record]") {
    int ngll = 2;
    int n_node_per_cell = ngll * ngll * ngll;  // 8
    int n_rec_cell = 2;
    int n_unique_gll = n_rec_cell * n_node_per_cell;  // 16 (no shared nodes)

    CompressionConfig comp;
    comp.method = CompressionMethod::None;

    // GLL-node recording map
    std::vector<int64_t> gll_node_ids(n_unique_gll);
    for (int i = 0; i < n_unique_gll; ++i)
        gll_node_ids[i] = i;
    std::vector<double> gll_node_coords(n_unique_gll * 3, 0.0);
    std::vector<int32_t> rec_cell_local = {0, 1};
    std::vector<int32_t> cell_gll_node_index(n_rec_cell * n_node_per_cell);
    for (int c = 0; c < n_rec_cell; ++c)
        for (int n = 0; n < n_node_per_cell; ++n)
            cell_gll_node_index[c * n_node_per_cell + n] = c * n_node_per_cell + n;

    RankData::RecordingMap rec_map;
    rec_map.has_recording = true;
    rec_map.gll_node_ids = gll_node_ids;
    rec_map.gll_node_coords = gll_node_coords;
    rec_map.rec_cell_local = rec_cell_local;
    rec_map.cell_gll_node_index = cell_gll_node_index;

    RecordWriter writer("./wavefields", "x", 0, rec_map, ngll, comp, false);

    // Write a few steps of strain data [n_rec_cell * n_node * 6]
    int strain_size = n_rec_cell * n_node_per_cell * 6;
    std::vector<double> strain(strain_size, 0.0);
    for (int step = 0; step < 3; ++step) {
        for (size_t i = 0; i < strain.size(); ++i) {
            strain[i] = static_cast<double>(step) * 1e-7;
        }
        writer.write_step(step, strain.data());
    }
    writer.close();

    // Verify per-step files with 4D strain [1, n_rec_cell, n_node, 6]
    {
        std::string fname0 = "./wavefields/x/record_0_0.h5";
        hid_t file0 = H5Fopen(fname0.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        REQUIRE(file0 >= 0);
        hid_t dset0 = H5Dopen2(file0, "strain", H5P_DEFAULT);
        REQUIRE(dset0 >= 0);
        hid_t space0 = H5Dget_space(dset0);
        REQUIRE(H5Sget_simple_extent_ndims(space0) == 4);
        hsize_t dims0[4];
        H5Sget_simple_extent_dims(space0, dims0, nullptr);
        REQUIRE(dims0[0] == 1);
        REQUIRE(dims0[1] == (hsize_t)n_rec_cell);
        REQUIRE(dims0[2] == (hsize_t)n_node_per_cell);
        REQUIRE(dims0[3] == 6);
        H5Sclose(space0);
        H5Dclose(dset0);
        // Verify gll_node_ids dataset exists
        hid_t id_dset = H5Dopen2(file0, "gll_node_ids", H5P_DEFAULT);
        REQUIRE(id_dset >= 0);
        H5Dclose(id_dset);
        H5Fclose(file0);
        std::remove(fname0.c_str());
    }
    std::remove("./wavefields/x/record_0_1.h5");
    std::remove("./wavefields/x/record_0_2.h5");
}

TEST_CASE("RecordWriter with float32", "[record]") {
    int ngll = 2;
    int n_node_per_cell = ngll * ngll * ngll;
    int n_rec_cell = 1;
    int n_unique_gll = n_rec_cell * n_node_per_cell;

    CompressionConfig comp;
    comp.method = CompressionMethod::None;

    std::vector<int64_t> gll_node_ids(n_unique_gll);
    for (int i = 0; i < n_unique_gll; ++i)
        gll_node_ids[i] = i + 10;
    std::vector<double> gll_node_coords(n_unique_gll * 3, 0.0);
    std::vector<int32_t> rec_cell_local = {0};
    std::vector<int32_t> cell_gll_node_index(n_node_per_cell);
    for (int n = 0; n < n_node_per_cell; ++n)
        cell_gll_node_index[n] = n;

    RankData::RecordingMap rec_map;
    rec_map.has_recording = true;
    rec_map.gll_node_ids = gll_node_ids;
    rec_map.gll_node_coords = gll_node_coords;
    rec_map.rec_cell_local = rec_cell_local;
    rec_map.cell_gll_node_index = cell_gll_node_index;

    RecordWriter writer("./wavefields", "y", 1, rec_map, ngll, comp, true);

    int strain_size = n_rec_cell * n_node_per_cell * 6;
    std::vector<double> strain(strain_size, 1e-6);
    writer.write_step(0, strain.data());
    writer.close();

    std::string fname = "./wavefields/y/record_1_0.h5";
    hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    H5Fclose(file);
    std::remove(fname.c_str());
}
