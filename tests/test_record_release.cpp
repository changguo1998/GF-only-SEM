// tests/test_record_release.cpp — production RecordWriter schema test
#include <hdf5.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "gf/record.hpp"

using namespace gf;

TEST_CASE("Release RecordWriter writes compact strain only", "[record][release]") {
    const std::string output_directory = "./wavefields";
    const std::string record_path = output_directory + "/release/record_7_4.h5";
    std::remove(record_path.c_str());

    RankData::RecordingMap recording_map;
    recording_map.rec_cell_local = {2, 0};
    constexpr int ngll = 2;
    constexpr int node_count = ngll * ngll * ngll;
    std::vector<double> strain(recording_map.rec_cell_local.size() * node_count * 6, 1.25);

    RecordWriter writer(output_directory, "release", 7, recording_map, ngll, 3, 2, false);
    writer.write_step(4, strain.data());

    hid_t file = H5Fopen(record_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    REQUIRE(H5Lexists(file, "strain", H5P_DEFAULT) > 0);
    REQUIRE(H5Lexists(file, "displacement", H5P_DEFAULT) == 0);
    REQUIRE(H5Lexists(file, "velocity", H5P_DEFAULT) == 0);
    REQUIRE(H5Lexists(file, "acceleration", H5P_DEFAULT) == 0);

    hid_t dataset = H5Dopen2(file, "strain", H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);
    hsize_t dimensions[4] = {0, 0, 0, 0};
    REQUIRE(H5Sget_simple_extent_ndims(dataspace) == 4);
    H5Sget_simple_extent_dims(dataspace, dimensions, nullptr);
    REQUIRE(dimensions[0] == 1);
    REQUIRE(dimensions[1] == recording_map.rec_cell_local.size());
    REQUIRE(dimensions[2] == node_count);
    REQUIRE(dimensions[3] == 6);
    hid_t creation_properties = H5Dget_create_plist(dataset);
    REQUIRE(H5Pget_nfilters(creation_properties) == 0);

    int stored_cell_count = -1;
    hid_t count_attribute = H5Aopen(file, "n_record_cell", H5P_DEFAULT);
    REQUIRE(count_attribute >= 0);
    REQUIRE(H5Aread(count_attribute, H5T_NATIVE_INT, &stored_cell_count) >= 0);
    REQUIRE(stored_cell_count == static_cast<int>(recording_map.rec_cell_local.size()));

    hid_t scope_attribute = H5Aopen(file, "cell_scope", H5P_DEFAULT);
    REQUIRE(scope_attribute >= 0);
    hid_t scope_type = H5Aget_type(scope_attribute);
    std::vector<char> scope(H5Tget_size(scope_type) + 1, '\0');
    REQUIRE(H5Aread(scope_attribute, scope_type, scope.data()) >= 0);
    REQUIRE(std::string(scope.data()) == "recording_cells");

    H5Tclose(scope_type);
    H5Aclose(scope_attribute);
    H5Aclose(count_attribute);
    H5Pclose(creation_properties);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(file);
    std::remove(record_path.c_str());
}
