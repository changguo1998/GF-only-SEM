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

// -----------------------------------------------------------------------
// Test cases
// -----------------------------------------------------------------------

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