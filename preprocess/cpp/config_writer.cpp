/// config_writer.cpp — write the config.h5 schema consumed by the forward solver

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "debug.hpp"
#include "gf_config.h"

namespace gf {
namespace {

hid_t create_config_file(const char* path) {
    hid_t file = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        fprintf(stderr, "ERROR: cannot create config.h5: %s\n", path);
        std::exit(1);
    }
    return file;
}

void write_int_attribute(hid_t location, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attribute = H5Acreate2(location, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, H5T_NATIVE_INT, &value);
    H5Aclose(attribute);
    H5Sclose(space);
}

void write_double_attribute(hid_t location, const char* name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attribute =
        H5Acreate2(location, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attribute);
    H5Sclose(space);
}

void write_string_attribute(hid_t location, const char* name, const std::string& value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, value.size() + 1);
    H5Tset_strpad(type, H5T_STR_NULLTERM);
    hid_t attribute = H5Acreate2(location, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, type, value.c_str());
    H5Aclose(attribute);
    H5Tclose(type);
    H5Sclose(space);
}

template <typename T>
void write_dataset(hid_t location, const char* name, const std::vector<T>& data, hid_t type,
                   const std::vector<hsize_t>& dimensions) {
    if (data.empty())
        return;
    hid_t space =
        H5Screate_simple(static_cast<int>(dimensions.size()), dimensions.data(), nullptr);
    hid_t dataset = H5Dcreate2(location, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(dataset);
    H5Sclose(space);
}

}  // namespace

void write_config_h5(const char* config_path, const Config& cfg, double solver_dt,
                     int snapshot_stride, int nsteps, const std::vector<double>& stf_t,
                     const std::vector<double>& stf_values, const SourceResult& source_result,
                     const double* domain_bounds, int nz_elements, double record_depth_actual_m) {
    GF_PREPROCESS_DEBUG_LOG("=== Writing config.h5 ===\n");
    hid_t config_file = create_config_file(config_path);

    // Simulation metadata uses the same schema as preprocess/config_writer.py.
    hid_t simulation =
        H5Gcreate2(config_file, "simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_string_attribute(simulation, "title", cfg.title);
    write_int_attribute(simulation, "polynomial_order", cfg.polynomial_order);
    write_double_attribute(simulation, "solver_dt", solver_dt);
    write_double_attribute(simulation, "output_dt_s", cfg.output_dt_s);
    write_int_attribute(simulation, "snapshot_stride", snapshot_stride);
    write_int_attribute(simulation, "nsteps", nsteps);
    write_double_attribute(simulation, "cfl_safety", cfg.cfl_safety);
    write_string_attribute(simulation, "snapshot_precision",
                           cfg.snapshot_precision_bytes == 4 ? "float32" : "float64");
    write_double_attribute(simulation, "storage_limit_gb", cfg.storage_limit_gb);
    write_double_attribute(simulation, "record_depth_max_m", cfg.record_depth_max_m);
    write_double_attribute(simulation, "record_depth_actual_m", record_depth_actual_m);
    write_int_attribute(simulation, "nx_elements", cfg.nx_elements);
    write_int_attribute(simulation, "ny_elements", cfg.ny_elements);
    write_int_attribute(simulation, "nz_elements", nz_elements);
    write_int_attribute(simulation, "pml_xmin", cfg.pml_xmin);
    write_int_attribute(simulation, "pml_xmax", cfg.pml_xmax);
    write_int_attribute(simulation, "pml_ymin", cfg.pml_ymin);
    write_int_attribute(simulation, "pml_ymax", cfg.pml_ymax);
    write_int_attribute(simulation, "pml_zmin", cfg.pml_zmin);
    write_int_attribute(simulation, "pml_zmax", cfg.pml_zmax);
    write_int_attribute(simulation, "n_ranks", cfg.n_ranks);
    write_int_attribute(simulation, "log_stride", cfg.log_stride);
    write_double_attribute(simulation, "restart_dt_s", cfg.restart_dt_s);
    int restart_stride =
        cfg.restart_dt_s > 0.0
            ? std::max(1, static_cast<int>(std::llround(cfg.restart_dt_s / solver_dt)))
            : 0;
    write_int_attribute(simulation, "restart_stride", restart_stride);
    std::vector<int64_t> tilex(cfg.tilex_elements.begin(), cfg.tilex_elements.end());
    std::vector<int64_t> tiley(cfg.tiley_elements.begin(), cfg.tiley_elements.end());
    write_dataset(simulation, "tilex_elements", tilex, H5T_NATIVE_INT64, {tilex.size()});
    write_dataset(simulation, "tiley_elements", tiley, H5T_NATIVE_INT64, {tiley.size()});
    H5Gclose(simulation);

    // Domain bounds are required by postprocess and visualization tools.
    hid_t domain = H5Gcreate2(config_file, "domain", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const char* bound_names[] = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"};
    for (int index = 0; index < 6; ++index)
        write_double_attribute(domain, bound_names[index], domain_bounds[index]);
    H5Gclose(domain);

    // Source samples and precomputed Lagrange weights are read directly by the solver.
    hid_t source = H5Gcreate2(config_file, "source", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_double_attribute(source, "x", cfg.source_x_m);
    write_double_attribute(source, "y", cfg.source_y_m);
    write_double_attribute(source, "z", cfg.source_z_m >= 0.0 ? cfg.source_z_m : domain_bounds[4]);
    write_double_attribute(source, "force_amplitude_n", cfg.source_force_amplitude_n);
    write_int_attribute(source, "n_src_cell", source_result.n_src_cell);
    write_dataset(source, "stf_t", stf_t, H5T_NATIVE_DOUBLE, {stf_t.size()});
    write_dataset(source, "stf_values", stf_values, H5T_NATIVE_DOUBLE, {stf_values.size()});
    if (source_result.n_src_cell > 0) {
        hid_t cells = H5Gcreate2(source, "cells", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        std::vector<int64_t> cell_ids(source_result.cell_ids.begin(),
                                      source_result.cell_ids.end());
        std::vector<double> weights;
        for (const auto& source_weights : source_result.weights)
            weights.insert(weights.end(), source_weights.begin(), source_weights.end());
        size_t weights_per_cell = source_result.weights.front().size();
        write_dataset(cells, "cell_ids", cell_ids, H5T_NATIVE_INT64, {cell_ids.size()});
        write_dataset(cells, "xi", source_result.xi, H5T_NATIVE_DOUBLE, {source_result.xi.size()});
        write_dataset(cells, "eta", source_result.eta, H5T_NATIVE_DOUBLE,
                      {source_result.eta.size()});
        write_dataset(cells, "zeta", source_result.zeta, H5T_NATIVE_DOUBLE,
                      {source_result.zeta.size()});
        write_dataset(cells, "weights", weights, H5T_NATIVE_DOUBLE,
                      {static_cast<hsize_t>(source_result.n_src_cell), weights_per_cell});
        H5Gclose(cells);
    }
    H5Gclose(source);

    H5Fclose(config_file);
    GF_PREPROCESS_DEBUG_LOG("  %s written\n", config_path);
}

}  // namespace gf
