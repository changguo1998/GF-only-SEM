// forward/src/io.cpp
#include "gf/io.hpp"
#include "gf/types.hpp"
#include <hdf5.h>
#include <stdexcept>
#include <iostream>

namespace gf {

namespace {

// Helper: open an HDF5 file for reading, close via RAII
struct H5FileGuard {
    hid_t id;
    explicit H5FileGuard(hid_t i) : id(i) {}
    ~H5FileGuard() { if (id >= 0) H5Fclose(id); }
    hid_t get() const { return id; }
};

// Open file, throw on failure
hid_t open_read(const std::string& path) {
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        throw std::runtime_error("H5Fopen failed: " + path);
    }
    return fid;
}

// Read a dataset given name, filling a typed vector
template <typename T>
std::vector<T> read_dataset_impl(hid_t file_id, const std::string& name) {
    hid_t dset = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
    if (dset < 0) {
        throw std::runtime_error("H5Dopen2 failed for: " + name);
    }
    H5FileGuard dset_guard(dset);

    hid_t dspace = H5Dget_space(dset);
    if (dspace < 0) {
        throw std::runtime_error("H5Dget_space failed for: " + name);
    }
    H5FileGuard space_guard(dspace);

    int ndims = H5Sget_simple_extent_ndims(dspace);
    if (ndims < 0) {
        throw std::runtime_error("H5Sget_simple_extent_ndims failed for: " + name);
    }

    hsize_t dims[8];
    H5Sget_simple_extent_dims(dspace, dims, nullptr);
    size_t total = 1;
    for (int i = 0; i < ndims; ++i) total *= dims[i];

    std::vector<T> data(total);
    // Determine native type
    hid_t nat_type;
    if (std::is_same<T, double>::value) {
        nat_type = H5T_NATIVE_DOUBLE;
    } else if (std::is_same<T, int64_t>::value) {
        nat_type = H5T_NATIVE_INT64;
    } else if (std::is_same<T, int32_t>::value) {
        nat_type = H5T_NATIVE_INT;
    } else {
        nat_type = H5T_NATIVE_DOUBLE;
    }

    herr_t status = H5Dread(dset, nat_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    if (status < 0) {
        throw std::runtime_error("H5Dread failed for: " + name);
    }
    return data;
}

} // anonymous namespace

// --- Public API implementations ---

std::vector<double> read_dataset_double(hid_t file_id, const std::string& name) {
    return read_dataset_impl<double>(file_id, name);
}

std::vector<int64_t> read_dataset_int64(hid_t file_id, const std::string& name) {
    return read_dataset_impl<int64_t>(file_id, name);
}

std::vector<int32_t> read_dataset_int32(hid_t file_id, const std::string& name) {
    return read_dataset_impl<int32_t>(file_id, name);
}

RankData read_partition(const std::string& path, int rank) {
    hid_t fid = open_read(path);
    H5FileGuard guard(fid);

    RankData data;

    // Read polynomial order from shape of coords or jacobian
    // Shape is typically [n_local_elem * NGLL^3, ...] so we can extract NGLL
    auto jacobian = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/jacobian");
    auto ngll = read_dataset_int32(fid, "/partition/rank_" + std::to_string(rank) + "/ngll");
    data.ngll = ngll.empty() ? 5 : static_cast<int>(ngll[0]); // default 5 if not found

    // Read element counts
    auto n_local = read_dataset_int64(fid, "/partition/rank_" + std::to_string(rank) + "/n_local_elem");
    auto n_ghost = read_dataset_int64(fid, "/partition/rank_" + std::to_string(rank) + "/n_ghost_elem");
    auto n_total = read_dataset_int64(fid, "/partition/rank_" + std::to_string(rank) + "/n_total_elem");
    data.n_local_elem = static_cast<int>(n_local[0]);
    data.n_ghost_elem = static_cast<int>(n_ghost[0]);
    data.n_total_elem = static_cast<int>(n_total[0]);

    // Read topology
    data.local_element_ids = read_dataset_int64(fid, "/partition/rank_" + std::to_string(rank) + "/local_element_ids");
    data.ghost_element_ids = read_dataset_int64(fid, "/partition/rank_" + std::to_string(rank) + "/ghost_element_ids");
    data.ghost_owners = read_dataset_int32(fid, "/partition/rank_" + std::to_string(rank) + "/ghost_owners");

    // Read geometry and material (precomputed at GLL nodes: [n_elem * NGLL^3, ...])
    data.coords  = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/coords");
    data.jacobian = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/jacobian");
    data.dxi_dx  = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/dxi_dx");
    data.mass    = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/mass");
    data.vp      = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/vp");
    data.vs      = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/vs");
    data.density = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/density");
    data.pml_damping = read_dataset_double(fid, "/partition/rank_" + std::to_string(rank) + "/pml_damping");

    // Read exchange patterns
    auto neighbors = read_dataset_int32(fid, "/partition/rank_" + std::to_string(rank) + "/neighbors");
    data.neighbors = neighbors;

    return data;
}

ConfigData read_config(const std::string& path) {
    hid_t fid = open_read(path);
    H5FileGuard guard(fid);

    ConfigData cfg;

    // Read attributes/datasets from config.h5 root group
    // Try as root-level datasets first, then under /config
    auto titles = read_dataset_double(fid, "title");
    if (titles.empty()) {
        cfg.title = "untitled";
    }

    cfg.polynomial_order = static_cast<int>(
        read_dataset_double(fid, "polynomial_order")[0]);

    cfg.dt = read_dataset_double(fid, "dt")[0];
    cfg.nsteps = static_cast<int>(read_dataset_double(fid, "nsteps")[0]);
    cfg.cfl_safety = read_dataset_double(fid, "cfl_safety")[0];
    cfg.checkpoint_interval = static_cast<int>(
        read_dataset_double(fid, "checkpoint_interval")[0]);
    cfg.checkpoint_precision = "float64";
    if (read_dataset_int64(fid, "use_float32")[0] == 1) {
        cfg.checkpoint_precision = "float32";
    }

    // Domain bounds
    auto xmin = read_dataset_double(fid, "xmin")[0];
    auto xmax = read_dataset_double(fid, "xmax")[0];
    auto ymin = read_dataset_double(fid, "ymin")[0];
    auto ymax = read_dataset_double(fid, "ymax")[0];
    auto zmin = read_dataset_double(fid, "zmin")[0];
    auto zmax = read_dataset_double(fid, "zmax")[0];
    cfg.xmin = xmin;  cfg.xmax = xmax;
    cfg.ymin = ymin;  cfg.ymax = ymax;
    cfg.zmin = zmin;  cfg.zmax = zmax;

    // Source: STFs and location
    cfg.stf_t    = read_dataset_double(fid, "stf_t");
    cfg.stf_values = read_dataset_double(fid, "stf_values");
    auto src_x = read_dataset_double(fid, "source_x");
    auto src_y = read_dataset_double(fid, "source_y");
    auto src_z = read_dataset_double(fid, "source_z");
    cfg.source_x = src_x[0];
    cfg.source_y = src_y[0];
    cfg.source_z = src_z[0];

    return cfg;
}

} // namespace gf