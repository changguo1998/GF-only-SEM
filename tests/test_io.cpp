// tests/test_io.cpp — I/O round-trip tests for partition/config readers
#include <hdf5.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gf/io.hpp"
#include "gf/types.hpp"

using namespace gf;
using Catch::Matchers::WithinAbs;

// Helper: create synthetic partition file
static std::string create_synth_partition(const std::string& path, int rank, int ngll,
                                          int n_local_cell) {
    // Use default file access for testing (no MPI I/O)
    hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(file >= 0);

    int n_node = n_local_cell * ngll * ngll * ngll;

    // Create /field/element group
    hid_t field_grp = H5Gcreate2(file, "/field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t elem_grp = H5Gcreate2(file, "/field/element", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // coords: [n_local_cell, NGLL, NGLL, NGLL, 3]
    hsize_t dims5[5] = {(hsize_t)n_local_cell, (hsize_t)ngll, (hsize_t)ngll, (hsize_t)ngll, 3};
    hid_t space5 = H5Screate_simple(5, dims5, nullptr);
    std::vector<double> coords(n_local_cell * ngll * ngll * ngll * 3, 0.0);
    hid_t dset = H5Dcreate2(elem_grp, "coords", H5T_NATIVE_DOUBLE, space5, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, coords.data());
    H5Dclose(dset);
    H5Sclose(space5);

    // jacobian: [n_local_cell, NGLL, NGLL, NGLL]
    hsize_t dims4[4] = {(hsize_t)n_local_cell, (hsize_t)ngll, (hsize_t)ngll, (hsize_t)ngll};
    hid_t space4 = H5Screate_simple(4, dims4, nullptr);
    std::vector<double> jac(n_node, 1.0);
    dset = H5Dcreate2(elem_grp, "jacobian", H5T_NATIVE_DOUBLE, space4, H5P_DEFAULT, H5P_DEFAULT,
                      H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, jac.data());
    H5Dclose(dset);
    H5Sclose(space4);

    // Write a few more fields
    std::vector<double> ones(n_node, 1.0);
    for (auto name : {"dxi_dx", "mass", "vp", "vs", "density", "damping"}) {
        hid_t s = H5Screate_simple(4, dims4, nullptr);
        std::vector<double> data(n_node, 1.0);
        hid_t d = H5Dcreate2(elem_grp, name, H5T_NATIVE_DOUBLE, s, H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT);
        H5Dwrite(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        H5Dclose(d);
        H5Sclose(s);
    }

    H5Gclose(elem_grp);
    H5Gclose(field_grp);

    // Create /partition group
    hid_t part_grp = H5Gcreate2(file, "/partition", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // local/ghost element ids
    hsize_t dims1[1] = {(hsize_t)n_local_cell};
    hid_t s1 = H5Screate_simple(1, dims1, nullptr);
    std::vector<int64_t> local_ids(n_local_cell);
    for (int i = 0; i < n_local_cell; ++i)
        local_ids[i] = rank * n_local_cell + i + 1;
    dset = H5Dcreate2(part_grp, "local_cell_ids", H5T_NATIVE_INT64, s1, H5P_DEFAULT, H5P_DEFAULT,
                      H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, local_ids.data());
    H5Dclose(dset);
    H5Sclose(s1);

    // Empty ghost ids
    hsize_t dims0[1] = {0};
    hid_t s0 = H5Screate_simple(1, dims0, nullptr);
    dset = H5Dcreate2(part_grp, "ghost_cell_ids", H5T_NATIVE_INT64, s0, H5P_DEFAULT, H5P_DEFAULT,
                      H5P_DEFAULT);
    H5Dclose(dset);
    H5Sclose(s0);

    s0 = H5Screate_simple(1, dims0, nullptr);
    dset = H5Dcreate2(part_grp, "ghost_owners", H5T_NATIVE_INT32, s0, H5P_DEFAULT, H5P_DEFAULT,
                      H5P_DEFAULT);
    H5Dclose(dset);
    H5Sclose(s0);

    // Exchange group
    hid_t exch_grp = H5Gcreate2(part_grp, "exchange", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    // Empty exchange for now (single rank, no neighbors)
    H5Gclose(exch_grp);
    H5Gclose(part_grp);
    H5Fclose(file);

    return path;
}

TEST_CASE("Read partition data round-trip", "[io]") {
    int ngll = 4;
    int n_local_cell = 2;
    std::string path = "test_partition_0.h5";
    create_synth_partition(path, 0, ngll, n_local_cell);

    RankData data = read_partition(path, 0);

    REQUIRE(data.n_local_cell == n_local_cell);
    REQUIRE(data.n_ghost_cell == 0);
    REQUIRE(data.ngll == ngll);
    REQUIRE(data.local_cell_ids.size() == static_cast<size_t>(n_local_cell));
    REQUIRE(data.ghost_cell_ids.empty());
    REQUIRE(data.ghost_owners.empty());

    // Cleanup
    std::remove(path.c_str());
}

TEST_CASE("Read config data round-trip", "[io]") {
    // Create a synthetic config.h5 using the new group/attribute schema.
    hid_t file = H5Fcreate("test_config.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(file >= 0);

    auto write_double_attr = [](hid_t loc, const char* name, double value) {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attr);
        H5Sclose(space);
    };
    auto write_int_attr = [](hid_t loc, const char* name, int value) {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &value);
        H5Aclose(attr);
        H5Sclose(space);
    };
    auto write_string_attr = [](hid_t loc, const char* name, const char* value) {
        hid_t type = H5Tcopy(H5T_C_S1);
        H5Tset_size(type, std::strlen(value));
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, type, value);
        H5Aclose(attr);
        H5Sclose(space);
        H5Tclose(type);
    };

    hid_t sim = H5Gcreate2(file, "/simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(sim >= 0);
    write_string_attr(sim, "title", "test");
    write_int_attr(sim, "polynomial_order", 3);
    write_double_attr(sim, "solver_dt", 0.001);
    write_double_attr(sim, "output_dt_s", 0.01);
    write_int_attr(sim, "snapshot_stride", 10);
    write_int_attr(sim, "nsteps", 500);
    write_double_attr(sim, "cfl_safety", 0.5);
    write_string_attr(sim, "snapshot_precision", "float32");
    H5Gclose(sim);

    hid_t domain = H5Gcreate2(file, "/domain", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(domain >= 0);
    write_double_attr(domain, "xmin", 0.0);
    write_double_attr(domain, "xmax", 1.0);
    write_double_attr(domain, "ymin", 0.0);
    write_double_attr(domain, "ymax", 2.0);
    write_double_attr(domain, "zmin", 0.0);
    write_double_attr(domain, "zmax", 3.0);
    H5Gclose(domain);

    hid_t source = H5Gcreate2(file, "/source", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    REQUIRE(source >= 0);
    write_double_attr(source, "x", 0.5);
    write_double_attr(source, "y", 0.6);
    write_double_attr(source, "z", 0.0);

    std::vector<double> stf(10, 1.0);
    hsize_t dims10[1] = {10};
    hid_t st = H5Screate_simple(1, dims10, nullptr);
    hid_t ds = H5Dcreate2(source, "stf_values", H5T_NATIVE_DOUBLE, st, H5P_DEFAULT, H5P_DEFAULT,
                          H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, stf.data());
    H5Dclose(ds);
    ds = H5Dcreate2(source, "stf_t", H5T_NATIVE_DOUBLE, st, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, stf.data());
    H5Dclose(ds);
    H5Sclose(st);
    H5Gclose(source);
    H5Fclose(file);

    ConfigData cfg = read_config("test_config.h5");
    REQUIRE(cfg.title == "test");
    REQUIRE(cfg.polynomial_order == 3);
    REQUIRE(cfg.solver_dt == 0.001);
    REQUIRE(cfg.output_dt_s == 0.01);
    REQUIRE(cfg.snapshot_stride == 10);
    REQUIRE(cfg.nsteps == 500);
    REQUIRE(cfg.snapshot_precision == "float32");
    REQUIRE(cfg.stf_values.size() == 10);
    REQUIRE(cfg.source_x == 0.5);

    std::remove("test_config.h5");
}