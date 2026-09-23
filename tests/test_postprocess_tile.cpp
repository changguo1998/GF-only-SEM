// test_postprocess_tile.cpp — Catch2 tests for greenfun tile schema
//
// Verifies that postprocess output tiles contain the expected self-contained
// attributes and datasets: source_xyz_m, vertex_coords, displacement_tensor.
//
// To run: requires that a tile has been generated (e.g. via examples/halfspace/postprocess.sh).
// Set the GF_TILE_PATH environment variable to the tile file, or this test
// will be skipped.

#include <hdf5.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"
#include "reader.hpp"

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static std::string get_tile_path() {
    const char* env = std::getenv("GF_TILE_PATH");
    if (env && env[0] != '\0')
        return std::string(env);
    return "";
}

static hid_t open_tile(const std::string& path) {
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0)
        throw std::runtime_error("Cannot open HDF5 file: " + path);
    return fid;
}

static bool attr_exists(hid_t loc, const char* name) {
    return H5Aexists(loc, name) > 0;
}

static bool dataset_exists(hid_t loc, const char* path) {
    return H5Lexists(loc, path, H5P_DEFAULT) > 0;
}

static std::string read_str_attr(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0)
        throw std::runtime_error(std::string("Cannot open attr: ") + name);
    hid_t atype = H5Aget_type(attr);
    hid_t stype = H5Tcopy(H5T_C_S1);
    H5Tset_size(stype, 256);
    char buf[256] = {0};
    H5Aread(attr, stype, buf);
    H5Tclose(stype);
    H5Tclose(atype);
    H5Aclose(attr);
    return std::string(buf);
}

static std::vector<hsize_t> get_dataset_dims(hid_t loc, const char* path) {
    hid_t ds = H5Dopen2(loc, path, H5P_DEFAULT);
    if (ds < 0)
        throw std::runtime_error(std::string("Cannot open dataset: ") + path);
    hid_t space = H5Dget_space(ds);
    int ndims = H5Sget_simple_extent_ndims(space);
    std::vector<hsize_t> dims((size_t)ndims);
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);
    H5Sclose(space);
    H5Dclose(ds);
    return dims;
}

static void write_record_field(hid_t file, const char* name, hsize_t cell_count,
                               hsize_t node_count, hsize_t component_count) {
    const hsize_t dimensions[4] = {1, cell_count, node_count, component_count};
    hid_t space = H5Screate_simple(4, dimensions, nullptr);
    hid_t dataset =
        H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> values((size_t)(cell_count * node_count * component_count));
    for (hsize_t cell = 0; cell < cell_count; ++cell) {
        for (hsize_t node = 0; node < node_count; ++node) {
            for (hsize_t component = 0; component < component_count; ++component) {
                values[(size_t)((cell * node_count + node) * component_count + component)] =
                    static_cast<double>(cell * 100 + node * 10 + component);
            }
        }
    }
    REQUIRE(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >=
            0);
    H5Dclose(dataset);
    H5Sclose(space);
}

// -----------------------------------------------------------------------
// Test cases
// -----------------------------------------------------------------------

TEST_CASE("postprocess reads selected cells from full-domain and legacy records",
          "[tile][record-reader]") {
    const std::string path = "./postprocess_record_reader_test.h5";
    std::remove(path.c_str());

    SECTION("full-domain selection preserves compact recording order") {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(file >= 0);
        const std::string scope = "all_local_cells";
        hid_t attribute_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(attribute_type, scope.size());
        hid_t attribute_space = H5Screate(H5S_SCALAR);
        hid_t attribute = H5Acreate2(file, "cell_scope", attribute_type, attribute_space,
                                     H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(H5Awrite(attribute, attribute_type, scope.c_str()) >= 0);
        H5Aclose(attribute);
        H5Sclose(attribute_space);
        H5Tclose(attribute_type);
        write_record_field(file, "strain", 4, 2, 6);
        write_record_field(file, "velocity", 4, 2, 3);
        H5Fclose(file);

        file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        const std::vector<int64_t> selected_cells = {3, 1};
        hsize_t cell_count = 0;
        hsize_t node_count = 0;
        std::vector<double> values;
        read_field_cells_4d(file, "strain", 6, selected_cells, cell_count, node_count, values);
        REQUIRE(cell_count == 2);
        REQUIRE(node_count == 2);
        REQUIRE(values.front() == 300.0);
        REQUIRE(values[12] == 100.0);
        read_field_cells_4d(file, "velocity", 3, selected_cells, cell_count, node_count, values);
        REQUIRE(values.front() == 300.0);
        REQUIRE(values[6] == 100.0);
        H5Fclose(file);
    }

    SECTION("legacy compact record is read directly") {
        hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        REQUIRE(file >= 0);
        write_record_field(file, "velocity", 2, 2, 3);
        H5Fclose(file);

        file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        const std::vector<int64_t> full_record_indices_not_present_in_legacy = {3, 1};
        hsize_t cell_count = 0;
        hsize_t node_count = 0;
        std::vector<double> values;
        read_field_cells_4d(file, "velocity", 3, full_record_indices_not_present_in_legacy,
                            cell_count, node_count, values);
        REQUIRE(cell_count == 2);
        REQUIRE(node_count == 2);
        REQUIRE(values.front() == 0.0);
        REQUIRE(values[6] == 100.0);
        H5Fclose(file);
    }

    std::remove(path.c_str());
}

TEST_CASE("postprocess strain tensor is component-major and direction-minor", "[tile]") {
    std::vector<double> greens_tensor(18, 0.0);
    const double strain_fx[] = {10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    const double strain_fy[] = {20.0, 21.0, 22.0, 23.0, 24.0, 25.0};
    const double strain_fz[] = {30.0, 31.0, 32.0, 33.0, 34.0, 35.0};

    gf_postprocess_common::assign_strain_direction(strain_fx, greens_tensor.data(), 0);
    gf_postprocess_common::assign_strain_direction(strain_fy, greens_tensor.data(), 1);
    gf_postprocess_common::assign_strain_direction(strain_fz, greens_tensor.data(), 2);

    for (int component = 0; component < 6; ++component) {
        CHECK(greens_tensor[component * 3 + 0] == strain_fx[component]);
        CHECK(greens_tensor[component * 3 + 1] == strain_fy[component]);
        CHECK(greens_tensor[component * 3 + 2] == strain_fz[component]);
    }
}

TEST_CASE("postprocess vector fields use independent averaging counts", "[tile]") {
    std::vector<double> displacement(3, 0.0);
    std::vector<double> velocity(3, 0.0);
    std::vector<double> acceleration(3, 0.0);
    const double displacement_first[] = {2.0, 4.0, 6.0};
    const double displacement_second[] = {4.0, 8.0, 12.0};
    const double velocity_sample[] = {10.0, 20.0, 30.0};
    const double acceleration_sample[] = {100.0, 200.0, 300.0};

    gf_postprocess_common::VectorFieldAverager displacement_average(displacement.data(), 1);
    gf_postprocess_common::VectorFieldAverager velocity_average(velocity.data(), 1);
    gf_postprocess_common::VectorFieldAverager acceleration_average(acceleration.data(), 1);

    displacement_average.add(0, displacement_first);
    displacement_average.add(0, displacement_second);
    velocity_average.add(0, velocity_sample);
    acceleration_average.add(0, acceleration_sample);
    displacement_average.normalize();
    velocity_average.normalize();
    acceleration_average.normalize();

    CHECK(displacement[0] == 3.0);
    CHECK(displacement[1] == 6.0);
    CHECK(displacement[2] == 9.0);
    CHECK(velocity[0] == 10.0);
    CHECK(acceleration[0] == 100.0);
}

TEST_CASE("postprocess tile assignment balances a shared record rank", "[tile]") {
    std::vector<gf_postprocess_common::TileAssignmentInput> tiles(8);
    for (auto& tile : tiles) {
        tile.work_weight = 100;
        tile.rank_read_weights = {{0, 1000}};
    }

    const auto assignments = gf_postprocess_common::assign_tiles_by_rank_overlap(tiles, 4);
    for (int worker = 0; worker < 4; ++worker) {
        CHECK(std::count(assignments.begin(), assignments.end(), worker) == 2);
    }
}

TEST_CASE("postprocess tile assignment reuses overlapping record ranks", "[tile]") {
    std::vector<gf_postprocess_common::TileAssignmentInput> tiles(4);
    for (auto& tile : tiles)
        tile.work_weight = 100;
    tiles[0].rank_read_weights = {{0, 1000}};
    tiles[1].rank_read_weights = {{1, 1000}};
    tiles[2].rank_read_weights = {{0, 1000}};
    tiles[3].rank_read_weights = {{1, 1000}};

    const auto assignments = gf_postprocess_common::assign_tiles_by_rank_overlap(tiles, 2);
    CHECK(assignments[0] == assignments[2]);
    CHECK(assignments[1] == assignments[3]);
    CHECK(assignments[0] != assignments[1]);
}

TEST_CASE("postprocess tile memory batches respect the worker budget", "[tile]") {
    const auto batches = gf_postprocess_common::plan_memory_batches({6, 6, 3, 12, 1}, 10);
    REQUIRE(batches.size() == 4);
    CHECK(batches[0].first == 0);
    CHECK(batches[0].second == 1);
    CHECK(batches[1].first == 1);
    CHECK(batches[1].second == 3);
    CHECK(batches[2].first == 3);
    CHECK(batches[2].second == 4);
    CHECK(batches[3].first == 4);
    CHECK(batches[3].second == 5);
}

TEST_CASE("postprocess memory budget uses the default or environment value", "[tile]") {
    constexpr uint64_t bytes_per_gib = 1024ULL * 1024 * 1024;
    CHECK(gf_postprocess_common::postprocess_memory_budget_bytes(nullptr) == 32 * bytes_per_gib);
    CHECK(gf_postprocess_common::postprocess_memory_budget_bytes("8") == 8 * bytes_per_gib);
}

TEST_CASE("postprocess memory budget rejects invalid environment values", "[tile]") {
    CHECK_THROWS_AS(gf_postprocess_common::postprocess_memory_budget_bytes(""),
                    std::invalid_argument);
    CHECK_THROWS_AS(gf_postprocess_common::postprocess_memory_budget_bytes("0"),
                    std::invalid_argument);
    CHECK_THROWS_AS(gf_postprocess_common::postprocess_memory_budget_bytes("1.5"),
                    std::invalid_argument);
    CHECK_THROWS_AS(gf_postprocess_common::postprocess_memory_budget_bytes("-1"),
                    std::invalid_argument);
    CHECK_THROWS_AS(gf_postprocess_common::postprocess_memory_budget_bytes("99999999999999999999"),
                    std::invalid_argument);
}

TEST_CASE("postprocess tile schema: source_xyz_m attribute", "[tile]") {
    auto path = get_tile_path();
    if (path.empty()) {
        WARN("GF_TILE_PATH not set — skipping test");
        return;
    }
    hid_t fid = open_tile(path);
    REQUIRE(attr_exists(fid, "source_xyz_m"));
    H5Fclose(fid);
}

TEST_CASE("postprocess tile schema: greens_quantities attribute", "[tile]") {
    auto path = get_tile_path();
    if (path.empty()) {
        WARN("GF_TILE_PATH not set — skipping test");
        return;
    }
    hid_t fid = open_tile(path);
    REQUIRE(attr_exists(fid, "greens_quantities"));
    std::string q = read_str_attr(fid, "greens_quantities");
    CHECK(q.find("strain") != std::string::npos);
    H5Fclose(fid);
}

TEST_CASE("postprocess tile schema: gll_node_coords dataset", "[tile]") {
    auto path = get_tile_path();
    if (path.empty()) {
        WARN("GF_TILE_PATH not set — skipping test");
        return;
    }
    hid_t fid = open_tile(path);
    REQUIRE(dataset_exists(fid, "/mesh/gll_node_coords"));
    auto dims = get_dataset_dims(fid, "/mesh/gll_node_coords");
    REQUIRE(dims.size() == 2);
    CHECK(dims[1] == 3);  // [n_local, 3]
    H5Fclose(fid);
}

TEST_CASE("postprocess tile schema: gll_node_ids dataset", "[tile]") {
    auto path = get_tile_path();
    if (path.empty()) {
        WARN("GF_TILE_PATH not set — skipping test");
        return;
    }
    hid_t fid = open_tile(path);
    REQUIRE(dataset_exists(fid, "/mesh/gll_node_ids"));
    auto dims = get_dataset_dims(fid, "/mesh/gll_node_ids");
    REQUIRE(dims.size() == 1);  // [n_local]
    H5Fclose(fid);
}

TEST_CASE("postprocess tile schema: greens_tensor dataset", "[tile]") {
    auto path = get_tile_path();
    if (path.empty()) {
        WARN("GF_TILE_PATH not set — skipping test");
        return;
    }
    hid_t fid = open_tile(path);
    REQUIRE(dataset_exists(fid, "/field/greens_tensor"));
    auto dims = get_dataset_dims(fid, "/field/greens_tensor");
    REQUIRE(dims.size() == 4);
    CHECK(dims[2] == 6);  // strain components
    CHECK(dims[3] == 3);  // force directions
    H5Fclose(fid);
}

TEST_CASE("postprocess tile schema: displacement_tensor dataset (if present)", "[tile]") {
    auto path = get_tile_path();
    if (path.empty()) {
        WARN("GF_TILE_PATH not set — skipping test");
        return;
    }
    hid_t fid = open_tile(path);
    std::string q = read_str_attr(fid, "greens_quantities");
    bool has_disp = (q.find("displacement") != std::string::npos);

    if (has_disp) {
        REQUIRE(dataset_exists(fid, "/field/displacement_tensor"));
        auto dims = get_dataset_dims(fid, "/field/displacement_tensor");
        REQUIRE(dims.size() == 4);
        CHECK(dims[2] == 3);  // displacement components
        CHECK(dims[3] == 3);  // force directions
    } else {
        WARN("Tile is strain-only, displacement_tensor not expected");
    }
    H5Fclose(fid);
}
