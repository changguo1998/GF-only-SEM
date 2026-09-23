// tests/test_record.cpp — RecordWriter tests (GLL-node format)
#include <hdf5.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gf/record.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

TEST_CASE("RecordWriter creates file and writes GLL strain", "[record]") {
    std::remove("./wavefields/layout_0.h5");
    int ngll = 2;
    int n_node_per_cell = ngll * ngll * ngll;  // 8
    int n_rec_cell = 2;
    std::vector<int32_t> rec_cell_local = {0, 1};

    RankData::RecordingMap rec_map;
    rec_map.has_recording = true;
    rec_map.rec_cell_local = rec_cell_local;

    RecordWriter writer("./wavefields", "x", 0, rec_map, ngll, 0, 1, false);

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

    REQUIRE_FALSE(std::filesystem::exists("./wavefields/layout_0.h5"));

    // Verify per-step files contain only dynamic fields and small identity attributes.
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
        REQUIRE(H5Lexists(file0, "gll_node_ids", H5P_DEFAULT) == 0);
        REQUIRE(H5Lexists(file0, "gll_node_coords", H5P_DEFAULT) == 0);
        REQUIRE(H5Lexists(file0, "cell_gll_node_index", H5P_DEFAULT) == 0);
        REQUIRE(H5Lexists(file0, "recording_cell_model_index", H5P_DEFAULT) == 0);
        int partition_start = -1;
        int partition_count = -1;
        hid_t start_attribute = H5Aopen(file0, "source_partition_start", H5P_DEFAULT);
        hid_t count_attribute = H5Aopen(file0, "source_partition_count", H5P_DEFAULT);
        REQUIRE(start_attribute >= 0);
        REQUIRE(count_attribute >= 0);
        REQUIRE(H5Aread(start_attribute, H5T_NATIVE_INT, &partition_start) >= 0);
        REQUIRE(H5Aread(count_attribute, H5T_NATIVE_INT, &partition_count) >= 0);
        REQUIRE(partition_start == 0);
        REQUIRE(partition_count == 1);
        H5Aclose(start_attribute);
        H5Aclose(count_attribute);
        H5Fclose(file0);
        std::remove(fname0.c_str());
    }
    std::remove("./wavefields/x/record_0_1.h5");
    std::remove("./wavefields/x/record_0_2.h5");
}

TEST_CASE("RecordWriter with float32", "[record]") {
    std::remove("./wavefields/layout_1.h5");
    int ngll = 2;
    int n_node_per_cell = ngll * ngll * ngll;
    int n_rec_cell = 1;
    std::vector<int32_t> rec_cell_local = {0};

    RankData::RecordingMap rec_map;
    rec_map.has_recording = true;
    rec_map.rec_cell_local = rec_cell_local;

    RecordWriter writer("./wavefields", "y", 1, rec_map, ngll, 1, 1, true);

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
