/// config_writer.cpp — write config.h5 metadata after preprocessing
///
/// Writes simulation parameters, mesh dimensions, PML configuration, source
/// location, STF arrays, and tiling info to a standalone config.h5 file.

#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gf_config.h"

namespace gf {
namespace {

// ── HDF5 helpers (config.h5 specific) ──────────────────────────────────────

hid_t create_config_h5(const char* path) {
    // Create new file (overwrite if exists)
    hid_t config_fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (config_fid < 0) {
        fprintf(stderr, "ERROR: cannot create config.h5: %s\n", path);
        std::exit(1);
    }
    return config_fid;
}

void write_double_dataset(hid_t fid, const char* name, const std::vector<double>& data,
                          hsize_t n) {
    hid_t space = H5Screate_simple(1, &n, nullptr);
    hid_t ds =
        H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(ds);
    H5Sclose(space);
}

/// Write an integer attribute to an HDF5 location.
void write_int_attr(hid_t loc, const char* name, int value) {
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    H5Sclose(attr_space);
}

/// Write a double attribute to an HDF5 location.
void write_double_attr(hid_t loc, const char* name, double value) {
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    H5Sclose(attr_space);
}

/// Write a string attribute to an HDF5 location.
void write_string_attr(hid_t loc, const char* name, const std::string& value) {
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, value.size());
    H5Tset_strpad(str_type, H5T_STR_NULLTERM);
    hid_t attr = H5Acreate2(loc, name, str_type, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, str_type, value.c_str());
    H5Aclose(attr);
    H5Tclose(str_type);
    H5Sclose(attr_space);
}

}  // namespace

void write_config_h5(const char* config_path, const Config& cfg, double solver_dt,
                     int snapshot_stride, int nsteps, const std::vector<double>& stf_t,
                     const std::vector<double>& stf_values, const std::vector<double>& source_xyz,
                     const SourceResult& src_result, double record_depth_actual_m,
                     const std::vector<int32_t>& element_to_rank, int n_ranks, double log_dt_s) {
    fprintf(stderr, "=== Writing config.h5 ===\n");

    hid_t config_fid = create_config_h5(config_path);

    // ── Simulation parameters ──
    hid_t sim_grp = H5Gcreate2(config_fid, "simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_double_attr(sim_grp, "solver_dt", solver_dt);
    write_double_attr(sim_grp, "output_dt_s", cfg.output_dt_s);
    write_double_attr(sim_grp, "total_duration_s", cfg.total_duration_s);
    write_int_attr(sim_grp, "nsteps", nsteps);
    write_int_attr(sim_grp, "snapshot_stride", snapshot_stride);
    write_int_attr(sim_grp, "log_stride", cfg.log_stride);
    write_double_attr(sim_grp, "cfl_safety", cfg.cfl_safety);
    write_int_attr(sim_grp, "polynomial_order", cfg.polynomial_order);
    write_int_attr(sim_grp, "n_ranks", n_ranks);
    write_double_attr(sim_grp, "storage_limit_gb", cfg.storage_limit_gb);
    write_double_attr(sim_grp, "record_depth_max_m", cfg.record_depth_max_m);
    write_double_attr(sim_grp, "record_depth_actual_m", record_depth_actual_m);
    write_double_attr(sim_grp, "restart_dt_s", cfg.restart_dt_s);
    write_int_attr(sim_grp, "snapshot_precision_bytes", cfg.snapshot_precision_bytes);
    write_double_attr(sim_grp, "log_dt_s", log_dt_s);
    H5Gclose(sim_grp);

    // ── Mesh parameters ──
    hid_t mesh_grp = H5Gcreate2(config_fid, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_int_attr(mesh_grp, "nx_elements", cfg.nx_elements);
    write_int_attr(mesh_grp, "ny_elements", cfg.ny_elements);
    write_double_attr(mesh_grp, "lx_m", cfg.lx_m);
    write_double_attr(mesh_grp, "ly_m", cfg.ly_m);
    write_double_attr(mesh_grp, "lz_m", cfg.lz_m);
    H5Gclose(mesh_grp);

    // ── PML ──
    hid_t pml_grp = H5Gcreate2(config_fid, "pml", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_int_attr(pml_grp, "xmin", cfg.pml_xmin);
    write_int_attr(pml_grp, "xmax", cfg.pml_xmax);
    write_int_attr(pml_grp, "ymin", cfg.pml_ymin);
    write_int_attr(pml_grp, "ymax", cfg.pml_ymax);
    write_int_attr(pml_grp, "zmin", cfg.pml_zmin);
    write_int_attr(pml_grp, "zmax", cfg.pml_zmax);
    write_double_attr(pml_grp, "f0_for_pml_hz", cfg.f0_for_pml_hz);
    H5Gclose(pml_grp);

    // ── Source ──
    hid_t src_grp = H5Gcreate2(config_fid, "source", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_double_attr(src_grp, "x_m", cfg.source_x_m);
    write_double_attr(src_grp, "y_m", cfg.source_y_m);
    write_double_attr(src_grp, "z_m", cfg.source_z_m);
    write_double_attr(src_grp, "force_amplitude_n", cfg.source_force_amplitude_n);
    H5Gclose(src_grp);

    // ── STF ──
    hsize_t n = static_cast<hsize_t>(stf_t.size());
    write_double_dataset(config_fid, "stf_time", stf_t, n);
    write_double_dataset(config_fid, "stf_values", stf_values, n);

    // ── Tiling (from config, if set) ──
    hsize_t ntx = static_cast<hsize_t>(cfg.tilex_elements.size());
    hsize_t nty = static_cast<hsize_t>(cfg.tiley_elements.size());
    if (ntx > 0) {
        std::vector<double> tilex_double(ntx);
        for (size_t i = 0; i < ntx; ++i)
            tilex_double[i] = static_cast<double>(cfg.tilex_elements[i]);
        write_double_dataset(config_fid, "tilex_elements", tilex_double, ntx);
    }
    if (nty > 0) {
        std::vector<double> tiley_double(nty);
        for (size_t i = 0; i < nty; ++i)
            tiley_double[i] = static_cast<double>(cfg.tiley_elements[i]);
        write_double_dataset(config_fid, "tiley_elements", tiley_double, nty);
    }

    // ── Title ──
    write_string_attr(config_fid, "title", cfg.title);

    H5Fclose(config_fid);
    fprintf(stderr, "  config.h5 written\n");
}

}  // namespace gf