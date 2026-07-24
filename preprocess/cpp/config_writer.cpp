/// config_writer.cpp — write config.h5 metadata after preprocessing
///
/// Writes solver_dt, snapshot_stride, nsteps, STF arrays, source info,
/// and domain bounds to config.h5.

#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gf_config.h"

namespace gf {
namespace {

hid_t create_or_open(const char* path) {
    // Create new file (overwrite if exists)
    hid_t fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "ERROR: cannot create config.h5: %s\n", path);
        std::exit(1);
    }
    return fid;
}

void write_double_vec(hid_t fid, const char* name, const std::vector<double>& data, hsize_t n) {
    hid_t space = H5Screate_simple(1, &n, nullptr);
    hid_t ds =
        H5Dcreate2(fid, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(ds);
    H5Sclose(space);
}

void write_int_attr(hid_t loc, const char* name, int value) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_INT, sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT, &value);
    H5Aclose(a);
    H5Sclose(sp);
}

void write_double_attr(hid_t loc, const char* name, double value) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(a);
    H5Sclose(sp);
}

void write_string_attr(hid_t loc, const char* name, const std::string& value) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t tp = H5Tcopy(H5T_C_S1);
    H5Tset_size(tp, value.size());
    H5Tset_strpad(tp, H5T_STR_NULLTERM);
    hid_t a = H5Acreate2(loc, name, tp, sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, tp, value.c_str());
    H5Aclose(a);
    H5Tclose(tp);
    H5Sclose(sp);
}

}  // namespace

void write_config_h5(const char* config_path, const gf::Config& cfg, double solver_dt,
                     int snapshot_stride, int nsteps, const std::vector<double>& stf_t,
                     const std::vector<double>& stf_values, const std::vector<double>& source_xyz,
                     const gf::SourceResult& src_result, double record_depth_actual_m,
                     const std::vector<int32_t>& element_to_rank, int n_ranks, double log_dt_s) {
    fprintf(stderr, "=== Writing config.h5 ===\n");

    hid_t fid = create_or_open(config_path);

    // ── Simulation parameters ──
    hid_t sim = H5Gcreate2(fid, "simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    write_double_attr(sim, "solver_dt", solver_dt);
    write_double_attr(sim, "output_dt_s", cfg.output_dt_s);
    write_double_attr(sim, "total_duration_s", cfg.total_duration_s);
    write_int_attr(sim, "nsteps", nsteps);
    write_int_attr(sim, "snapshot_stride", snapshot_stride);
    write_int_attr(sim, "log_stride", cfg.log_stride);
    write_double_attr(sim, "cfl_safety", cfg.cfl_safety);
    write_int_attr(sim, "polynomial_order", cfg.polynomial_order);
    write_int_attr(sim, "n_ranks", n_ranks);
    write_double_attr(sim, "storage_limit_gb", cfg.storage_limit_gb);
    write_double_attr(sim, "record_depth_max_m", cfg.record_depth_max_m);
    write_double_attr(sim, "record_depth_actual_m", record_depth_actual_m);
    write_double_attr(sim, "restart_dt_s", cfg.restart_dt_s);
    write_int_attr(sim, "snapshot_precision_bytes", cfg.snapshot_precision_bytes);
    write_double_attr(sim, "log_dt_s", log_dt_s);

    H5Gclose(sim);

    // ── Mesh parameters ──
    hid_t mesh = H5Gcreate2(fid, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_int_attr(mesh, "nx_elements", cfg.nx_elements);
    write_int_attr(mesh, "ny_elements", cfg.ny_elements);
    write_double_attr(mesh, "lx_m", cfg.lx_m);
    write_double_attr(mesh, "ly_m", cfg.ly_m);
    write_double_attr(mesh, "lz_m", cfg.lz_m);
    H5Gclose(mesh);

    // ── PML ──
    hid_t pml = H5Gcreate2(fid, "pml", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_int_attr(pml, "xmin", cfg.pml_xmin);
    write_int_attr(pml, "xmax", cfg.pml_xmax);
    write_int_attr(pml, "ymin", cfg.pml_ymin);
    write_int_attr(pml, "ymax", cfg.pml_ymax);
    write_int_attr(pml, "zmin", cfg.pml_zmin);
    write_int_attr(pml, "zmax", cfg.pml_zmax);
    write_double_attr(pml, "f0_for_pml_hz", cfg.f0_for_pml_hz);
    H5Gclose(pml);

    // ── Source ──
    hid_t src = H5Gcreate2(fid, "source", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_double_attr(src, "x_m", cfg.source_x_m);
    write_double_attr(src, "y_m", cfg.source_y_m);
    write_double_attr(src, "z_m", cfg.source_z_m);
    write_double_attr(src, "force_amplitude_n", cfg.source_force_amplitude_n);
    H5Gclose(src);

    // ── STF ──
    hsize_t n = static_cast<hsize_t>(stf_t.size());
    write_double_vec(fid, "stf_time", stf_t, n);
    write_double_vec(fid, "stf_values", stf_values, n);

    // ── Tiling (tilex/tiley from config) ──
    hsize_t ntx = static_cast<hsize_t>(cfg.tilex_elements.size());
    hsize_t nty = static_cast<hsize_t>(cfg.tiley_elements.size());
    if (ntx > 0) {
        std::vector<double> tx(ntx);
        for (size_t i = 0; i < ntx; ++i)
            tx[i] = static_cast<double>(cfg.tilex_elements[i]);
        write_double_vec(fid, "tilex_elements", tx, ntx);
    }
    if (nty > 0) {
        std::vector<double> ty(nty);
        for (size_t i = 0; i < nty; ++i)
            ty[i] = static_cast<double>(cfg.tiley_elements[i]);
        write_double_vec(fid, "tiley_elements", ty, nty);
    }

    // ── Title ──
    write_string_attr(fid, "title", cfg.title);

    H5Fclose(fid);
    fprintf(stderr, "  config.h5 written\n");
}

}  // namespace gf