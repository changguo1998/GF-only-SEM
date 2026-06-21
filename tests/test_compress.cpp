// tests/test_compress.cpp — HDF5 compression round-trip tests
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <hdf5.h>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gf/CompressionFilter.h"
#include "gf/PrecisionPolicy.h"
#include "gf/ChunkingStrategy.h"
#include "gf/CheckpointWriter.h"

using namespace gf;
using Catch::Matchers::WithinAbs;

// Helper: write a checkpoint file and read it back, comparing values
static void roundtrip_test(const CompressionConfig& comp, bool use_float32,
                           double abs_tol)
{
    constexpr int ngll = 4;        // N=3
    constexpr hsize_t num_elems = 8;
    constexpr hsize_t n_node = ngll * ngll * ngll;
    constexpr hsize_t total = 6 * n_node * num_elems;

    // Generate synthetic strain data (smooth spatially varying field)
    std::vector<double> original(total);
    for (hsize_t i = 0; i < total; ++i) {
        original[i] = std::sin(static_cast<double>(i) * 0.1) * 1e-6;
    }

    std::string filename = "test_roundtrip.h5";
    std::remove(filename.c_str());

    // Create file
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC,
                              H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(file_id >= 0);

    // Write checkpoint
    CheckpointConfig cfg;
    cfg.compression = comp;
    cfg.use_float32 = use_float32;
    cfg.ngll = ngll;

    hid_t dset_id = write_checkpoint(file_id, num_elems,
                                     nullptr, cfg);
    REQUIRE(dset_id >= 0);

    // Extend and write data at step 0
    hsize_t size[6] = {1, num_elems,
                       static_cast<hsize_t>(ngll),
                       static_cast<hsize_t>(ngll),
                       static_cast<hsize_t>(ngll), 6};
    H5Dextend(dset_id, size);

    hid_t filespace = H5Dget_space(dset_id);
    hsize_t start[6] = {0, 0, 0, 0, 0, 0};
    hsize_t count[6] = {1, num_elems,
                        static_cast<hsize_t>(ngll),
                        static_cast<hsize_t>(ngll),
                        static_cast<hsize_t>(ngll), 6};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

    hid_t memspace = H5Screate_simple(6, size, nullptr);
    hid_t write_type = select_precision_type(cfg.use_float32);
    H5Dwrite(dset_id, write_type, memspace, filespace, H5P_DEFAULT, original.data());

    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dset_id);
    H5Fclose(file_id);

    // Reopen and read back
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file_id >= 0);

    dset_id = H5Dopen2(file_id, "strain", H5P_DEFAULT);
    REQUIRE(dset_id >= 0);

    // Read back as float64
    std::vector<double> roundtrip(total);
    herr_t status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, roundtrip.data());
    REQUIRE(status >= 0);

    H5Dclose(dset_id);
    H5Fclose(file_id);

    // Compare
    for (hsize_t i = 0; i < total; ++i) {
        REQUIRE_THAT(roundtrip[i],
                     Catch::Matchers::WithinAbs(original[i], abs_tol));
    }

    // Cleanup
    std::remove(filename.c_str());
}

TEST_CASE("No compression round-trip (float64)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::None;
    // float64 with no compression should be bit-exact
    roundtrip_test(cfg, false, 0.0);
}

TEST_CASE("Zlib level 1 round-trip (float64)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::Zlib;
    cfg.zlib.level = 1;
    // Zlib is lossless, so bit-exact for same type
    roundtrip_test(cfg, false, 0.0);
}

TEST_CASE("Zlib level 6 round-trip (float64)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::Zlib;
    cfg.zlib.level = 6;
    roundtrip_test(cfg, false, 0.0);
}

TEST_CASE("Zlib level 9 round-trip (float64)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::Zlib;
    cfg.zlib.level = 9;
    roundtrip_test(cfg, false, 0.0);
}

TEST_CASE("LZF round-trip (float64)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::LZF;
    // LZF is lossless, so bit-exact
    roundtrip_test(cfg, false, 0.0);
}

TEST_CASE("Zlib level 6 round-trip (float32)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::Zlib;
    cfg.zlib.level = 6;
    // float32 loses precision vs float64: ~1e-7 relative tolerance
    roundtrip_test(cfg, true, 1e-6);
}

TEST_CASE("Zlib level 1 round-trip (float32)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::Zlib;
    cfg.zlib.level = 1;
    roundtrip_test(cfg, true, 1e-6);
}

TEST_CASE("LZF round-trip (float32)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::LZF;
    roundtrip_test(cfg, true, 1e-6);
}

TEST_CASE("No compression round-trip (float32)", "[compress]") {
    CompressionConfig cfg;
    cfg.method = CompressionMethod::None;
    roundtrip_test(cfg, true, 1e-6);
}