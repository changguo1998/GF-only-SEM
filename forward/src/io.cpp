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

// Try-read a dataset; return empty vector if not found
template <typename T>
std::vector<T> try_read_dataset(hid_t file_id, const std::string& name) {
    if (H5Lexists(file_id, name.c_str(), H5P_DEFAULT) > 0) {
        return read_dataset_impl<T>(file_id, name);
    }
    return {};
}

// Read int32 attribute
bool read_attr_int(hid_t loc_id, const std::string& name, int& out) {
    if (H5Aexists(loc_id, name.c_str()) > 0) {
        hid_t attr = H5Aopen(loc_id, name.c_str(), H5P_DEFAULT);
        H5FileGuard attr_guard(attr);
        if (attr >= 0) {
            H5Aread(attr, H5T_NATIVE_INT, &out);
            return true;
        }
    }
    return false;
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

    // --- Read element counts ---
    auto local_ids = read_dataset_int64(fid, "/partition/local_element_ids");
    auto ghost_ids = read_dataset_int64(fid, "/partition/ghost_element_ids");
    auto ghost_owners = read_dataset_int32(fid, "/partition/ghost_owners");

    data.n_local_elem = static_cast<int>(local_ids.size());
    data.n_ghost_elem = static_cast<int>(ghost_ids.size());
    data.n_total_elem = data.n_local_elem + data.n_ghost_elem;

    data.local_element_ids = local_ids;
    data.ghost_element_ids = ghost_ids;
    data.ghost_owners = ghost_owners;

    // --- Derive NGLL from array shape ---
    // coords has shape [n_elem, NGLL, NGLL, NGLL, 3]
    {
        hid_t dset = H5Dopen2(fid, "/field/element/coords", H5P_DEFAULT);
        hid_t dspace = H5Dget_space(dset);
        int ndims = H5Sget_simple_extent_ndims(dspace);
        hsize_t dims[8];
        H5Sget_simple_extent_dims(dspace, dims, nullptr);
        if (ndims >= 4) {
            data.ngll = static_cast<int>(dims[1]);  // NGLL = shape[1]
        } else {
            data.ngll = 5;  // default N+1=5
        }
        H5Sclose(dspace);
        H5Dclose(dset);
    }

    // --- Read geometry and material fields ---
    // All stored under /field/element/ with shape [n_local_elem, NGLL, NGLL, NGLL, ...]
    data.coords    = try_read_dataset<double>(fid, "/field/element/coords");
    data.jacobian  = try_read_dataset<double>(fid, "/field/element/jacobian");
    data.dxi_dx    = try_read_dataset<double>(fid, "/field/element/dxi_dx");
    data.mass      = try_read_dataset<double>(fid, "/field/element/mass");
    data.vp        = try_read_dataset<double>(fid, "/field/element/vp");
    data.vs        = try_read_dataset<double>(fid, "/field/element/vs");
    data.density   = try_read_dataset<double>(fid, "/field/element/density");
    data.pml_damping = try_read_dataset<double>(fid, "/field/element/damping");

    // --- Read exchange patterns ---
    hid_t exch_grp = H5Gopen2(fid, "/partition/exchange", H5P_DEFAULT);
    if (exch_grp >= 0) {
        H5FileGuard exch_guard(exch_grp);

        hsize_t num_neighbors = 0;
        H5G_info_t grp_info;
        herr_t info_status = H5Gget_info(exch_grp, &grp_info);
        if (info_status >= 0) {
            num_neighbors = grp_info.nlinks;
        }

        for (hsize_t i = 0; i < num_neighbors; ++i) {
            char link_name[256];
            ssize_t name_len = H5Lget_name_by_idx(
                exch_grp, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
                i, link_name, sizeof(link_name), H5P_DEFAULT);

            if (name_len <= 0) continue;

            std::string neighbor_name(link_name, name_len);
            // neighbor_name is like "neighbor_1"
            // Extract rank number after underscore
            size_t underscore = neighbor_name.find('_');
            if (underscore == std::string::npos) continue;

            std::string rank_str = neighbor_name.substr(underscore + 1);
            int neighbor_rank = std::stoi(rank_str);

            hid_t ng = H5Gopen2(exch_grp, neighbor_name.c_str(), H5P_DEFAULT);
            if (ng < 0) continue;
            H5FileGuard ng_guard(ng);

            auto send_dof = try_read_dataset<int32_t>(ng, "send_dof");
            auto recv_dof = try_read_dataset<int32_t>(ng, "recv_dof");

            if (!send_dof.empty() || !recv_dof.empty()) {
                RankData::ExchangePattern pat;
                pat.neighbor_rank = neighbor_rank;
                pat.send_dof_indices.assign(send_dof.begin(), send_dof.end());
                pat.recv_dof_indices.assign(recv_dof.begin(), recv_dof.end());
                data.exchange_patterns.push_back(std::move(pat));
            }
        }
    }

    return data;
}

ConfigData read_config(const std::string& path) {
    hid_t fid = open_read(path);
    H5FileGuard guard(fid);

    ConfigData cfg;

    cfg.title = "untitled";

    auto poly_order = try_read_dataset<double>(fid, "polynomial_order");
    cfg.polynomial_order = poly_order.empty() ? 3 : static_cast<int>(poly_order[0]);

    auto dt = try_read_dataset<double>(fid, "dt");
    cfg.dt = dt.empty() ? 0.005 : dt[0];

    auto nsteps = try_read_dataset<double>(fid, "nsteps");
    cfg.nsteps = nsteps.empty() ? 1000 : static_cast<int>(nsteps[0]);

    auto cfl = try_read_dataset<double>(fid, "cfl_safety");
    cfg.cfl_safety = cfl.empty() ? 1.0 : cfl[0];

    auto ckpt_int = try_read_dataset<double>(fid, "checkpoint_interval");
    cfg.checkpoint_interval = ckpt_int.empty() ? 10 : static_cast<int>(ckpt_int[0]);

    cfg.checkpoint_precision = "float64";
    auto use_f32 = try_read_dataset<int64_t>(fid, "use_float32");
    if (!use_f32.empty() && use_f32[0] == 1) {
        cfg.checkpoint_precision = "float32";
    }

    // Domain bounds
    auto xmin = try_read_dataset<double>(fid, "xmin");
    auto xmax = try_read_dataset<double>(fid, "xmax");
    auto ymin = try_read_dataset<double>(fid, "ymin");
    auto ymax = try_read_dataset<double>(fid, "ymax");
    auto zmin = try_read_dataset<double>(fid, "zmin");
    auto zmax = try_read_dataset<double>(fid, "zmax");
    if (!xmin.empty()) cfg.xmin = xmin[0];
    if (!xmax.empty()) cfg.xmax = xmax[0];
    if (!ymin.empty()) cfg.ymin = ymin[0];
    if (!ymax.empty()) cfg.ymax = ymax[0];
    if (!zmin.empty()) cfg.zmin = zmin[0];
    if (!zmax.empty()) cfg.zmax = zmax[0];

    // Source data
    cfg.stf_t      = try_read_dataset<double>(fid, "stf_t");
    cfg.stf_values = try_read_dataset<double>(fid, "stf_values");
    auto src_x = try_read_dataset<double>(fid, "source_x");
    auto src_y = try_read_dataset<double>(fid, "source_y");
    auto src_z = try_read_dataset<double>(fid, "source_z");
    if (!src_x.empty()) cfg.source_x = src_x[0];
    if (!src_y.empty()) cfg.source_y = src_y[0];
    if (!src_z.empty()) cfg.source_z = src_z[0];

    return cfg;
}

} // namespace gf