// forward/share/src/io.cpp
#include "gf/io.hpp"

#include <hdf5.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "gf/types.hpp"

namespace gf {

namespace {

// Helper: close HDF5 identifiers via RAII.
struct H5FileGuard {
    hid_t id;
    explicit H5FileGuard(hid_t i) : id(i) {}
    ~H5FileGuard() {
        if (id < 0)
            return;
        H5I_type_t type = H5Iget_type(id);
        if (type == H5I_FILE)
            H5Fclose(id);
        else if (type == H5I_GROUP)
            H5Gclose(id);
        else if (type == H5I_DATASET)
            H5Dclose(id);
        else if (type == H5I_DATASPACE)
            H5Sclose(id);
        else if (type == H5I_ATTR)
            H5Aclose(id);
        else if (type == H5I_DATATYPE)
            H5Tclose(id);
    }
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
    for (int i = 0; i < ndims; ++i)
        total *= dims[i];

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

bool read_attr_int(hid_t loc_id, const std::string& name, int& out) {
    if (H5Aexists(loc_id, name.c_str()) <= 0)
        return false;
    hid_t attr = H5Aopen(loc_id, name.c_str(), H5P_DEFAULT);
    if (attr < 0)
        return false;
    H5FileGuard attr_guard(attr);
    return H5Aread(attr, H5T_NATIVE_INT, &out) >= 0;
}

bool read_attr_double(hid_t loc_id, const std::string& name, double& out) {
    if (H5Aexists(loc_id, name.c_str()) <= 0)
        return false;
    hid_t attr = H5Aopen(loc_id, name.c_str(), H5P_DEFAULT);
    if (attr < 0)
        return false;
    H5FileGuard attr_guard(attr);
    return H5Aread(attr, H5T_NATIVE_DOUBLE, &out) >= 0;
}

bool read_attr_string(hid_t loc_id, const std::string& name, std::string& out) {
    if (H5Aexists(loc_id, name.c_str()) <= 0)
        return false;
    hid_t attr = H5Aopen(loc_id, name.c_str(), H5P_DEFAULT);
    if (attr < 0)
        return false;
    H5FileGuard attr_guard(attr);

    hid_t type = H5Aget_type(attr);
    if (type < 0)
        return false;
    H5FileGuard type_guard(type);

    if (H5Tis_variable_str(type) > 0) {
        char* value = nullptr;
        if (H5Aread(attr, type, &value) < 0)
            return false;
        out = value ? std::string(value) : std::string();
        if (value)
            H5free_memory(value);
        return true;
    }

    size_t len = H5Tget_size(type);
    std::vector<char> buf(len + 1, '\0');
    if (H5Aread(attr, type, buf.data()) < 0)
        return false;
    out = std::string(buf.data());
    return true;
}

}  // anonymous namespace

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

std::vector<int> read_dataset_int(hid_t file_id, const std::string& name) {
    // Check if dataset exists; return empty if not.
    if (H5Lexists(file_id, name.c_str(), H5P_DEFAULT) <= 0) {
        return {};
    }
    auto tmp = read_dataset_impl<int64_t>(file_id, name);
    return std::vector<int>(tmp.begin(), tmp.end());
}

RankData read_partition(const std::string& path, int /*rank*/) {
    hid_t fid = open_read(path);
    H5FileGuard guard(fid);

    RankData data;

    // --- Read element counts ---
    auto local_ids = read_dataset_int64(fid, "/partition/local_cell_ids");
    auto ghost_ids = try_read_dataset<int64_t>(fid, "/partition/ghost_cell_ids");
    auto ghost_owners = try_read_dataset<int32_t>(fid, "/partition/ghost_owners");

    data.n_local_cell = static_cast<int>(local_ids.size());
    data.n_ghost_cell = static_cast<int>(ghost_ids.size());
    data.n_total_cell = data.n_local_cell + data.n_ghost_cell;

    data.local_cell_ids = local_ids;
    data.ghost_cell_ids = ghost_ids;
    data.ghost_owners = ghost_owners;

    // --- Derive NGLL from array shape ---
    // coords has shape [n_elem, NGLL, NGLL, NGLL, 3]
    {
        hid_t dset = H5Dopen2(fid, "/field/cell/coords", H5P_DEFAULT);
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
    // All stored under /field/cell/ with shape [n_local_cell, NGLL, NGLL, NGLL, ...]
    data.coords = try_read_dataset<double>(fid, "/field/cell/coords");
    data.jacobian = try_read_dataset<double>(fid, "/field/cell/jacobian");
    data.dxi_dx = try_read_dataset<double>(fid, "/field/cell/dxi_dx");
    data.mass = try_read_dataset<double>(fid, "/field/cell/mass");
    data.vp = try_read_dataset<double>(fid, "/field/cell/vp");
    data.vs = try_read_dataset<double>(fid, "/field/cell/vs");
    data.density = try_read_dataset<double>(fid, "/field/cell/density");
    data.lambda_ = try_read_dataset<double>(fid, "/field/cell/lambda");
    data.mu_ = try_read_dataset<double>(fid, "/field/cell/mu");
    data.pml_damping = try_read_dataset<double>(fid, "/field/cell/damping");

    // --- Read local_cell2rank_node and n_rank_node (CG-SEM rank-level node mapping) ---
    data.local_cell2rank_node = try_read_dataset<int32_t>(fid, "/field/cell/local_cell2rank_node");
    data.local_cell2global_node =
        try_read_dataset<int32_t>(fid, "/field/cell/local_cell2global_node");
    {
        hid_t felem = H5Gopen2(fid, "/field/cell", H5P_DEFAULT);
        if (felem >= 0) {
            read_attr_int(felem, "n_rank_node", data.n_rank_node);
            H5Gclose(felem);
        }
    }

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
            ssize_t name_len = H5Lget_name_by_idx(exch_grp, ".", H5_INDEX_NAME, H5_ITER_NATIVE, i,
                                                  link_name, sizeof(link_name), H5P_DEFAULT);

            if (name_len <= 0)
                continue;

            std::string neighbor_name(link_name, name_len);
            // neighbor_name is like "neighbor_1"
            // Extract rank number after underscore
            size_t underscore = neighbor_name.find('_');
            if (underscore == std::string::npos)
                continue;

            std::string rank_str = neighbor_name.substr(underscore + 1);
            int neighbor_rank = std::stoi(rank_str);

            hid_t ng = H5Gopen2(exch_grp, neighbor_name.c_str(), H5P_DEFAULT);
            if (ng < 0)
                continue;
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

    // --- Read recording map ---
    hid_t rec_grp = -1;
    H5E_BEGIN_TRY {
        rec_grp = H5Gopen2(fid, "/recording", H5P_DEFAULT);
    }
    H5E_END_TRY;
    if (rec_grp >= 0) {
        H5FileGuard rec_guard(rec_grp);
        data.recording.has_recording = true;
        data.recording.vertex_ids = read_dataset_int64(fid, "/recording/vertex_ids");
        data.recording.src_elem_local =
            read_dataset_int32(fid, "/recording/source_element_local_index");
        // Read corner index as int32
        data.recording.src_corner = read_dataset_int32(fid, "/recording/source_corner_index");
    }

    return data;
}

RankData read_partition_all(const std::string& partition_dir) {
    // Count partitions by scanning partition_{r}.h5 files
    int n_partitions = 0;
    for (int r = 0;; ++r) {
        std::string path = partition_dir + "/partition_" + std::to_string(r) + ".h5";
        std::error_code ec;
        if (!std::filesystem::exists(path, ec))
            break;
        ++n_partitions;
    }

    if (n_partitions == 0) {
        throw std::runtime_error("read_partition_all: no partition files found in " +
                                 partition_dir);
    }

    // Helper: append src to dest
    auto concat_vec = [](auto& dest, const auto& src) {
        dest.insert(dest.end(), src.begin(), src.end());
    };

    RankData merged;
    int cumulative_elements = 0;

    for (int r = 0; r < n_partitions; ++r) {
        std::string path = partition_dir + "/partition_" + std::to_string(r) + ".h5";
        RankData part = read_partition(path, r);

        // Verify NGLL consistency
        if (r > 0 && part.ngll != merged.ngll) {
            throw std::runtime_error("read_partition_all: partition " + std::to_string(r) +
                                     " has NGLL=" + std::to_string(part.ngll) + " but expected " +
                                     std::to_string(merged.ngll));
        }

        // Use local_cell2global_node for merged ibool — global IDs are already
        // consistent across ranks (shared nodes have same global ID), so no offset
        // is needed.  This correctly couples elements across original rank
        // boundaries for single-GPU CG-SEM.
        if (r == 0) {
            // Save global ibool before move (part is consumed by std::move below)
            auto global_ibool = std::move(part.local_cell2global_node);
            merged = std::move(part);
            merged.exchange_patterns.clear();
            merged.ghost_cell_ids.clear();
            merged.ghost_owners.clear();
            merged.local_cell2rank_node = std::move(global_ibool);
            cumulative_elements = merged.n_local_cell;
        } else {
            // Concatenate element-based arrays
            concat_vec(merged.local_cell_ids, part.local_cell_ids);
            concat_vec(merged.coords, part.coords);
            concat_vec(merged.jacobian, part.jacobian);
            concat_vec(merged.dxi_dx, part.dxi_dx);
            concat_vec(merged.mass, part.mass);
            concat_vec(merged.vp, part.vp);
            concat_vec(merged.vs, part.vs);
            concat_vec(merged.density, part.density);
            concat_vec(merged.lambda_, part.lambda_);
            concat_vec(merged.mu_, part.mu_);
            concat_vec(merged.pml_damping, part.pml_damping);

            // Merge ibool using global node IDs (already unique across ranks)
            concat_vec(merged.local_cell2rank_node, part.local_cell2global_node);

            // Merge recording: adjust src_elem_local by cumulative element count
            if (part.recording.has_recording) {
                merged.recording.has_recording = true;
                concat_vec(merged.recording.vertex_ids, part.recording.vertex_ids);
                for (int32_t local_idx : part.recording.src_elem_local) {
                    merged.recording.src_elem_local.push_back(local_idx + cumulative_elements);
                }
                concat_vec(merged.recording.src_corner, part.recording.src_corner);
            }

            cumulative_elements += part.n_local_cell;
        }
    }

    merged.n_local_cell = cumulative_elements;
    merged.n_ghost_cell = 0;
    merged.n_total_cell = merged.n_local_cell;

    // n_rank_node = max global node ID + 1 (global IDs are 0..n_global_node-1)
    if (!merged.local_cell2rank_node.empty()) {
        int max_id = *std::max_element(merged.local_cell2rank_node.begin(),
                                       merged.local_cell2rank_node.end());
        merged.n_rank_node = max_id + 1;
    }

    return merged;
}

RankData read_partition_range(const std::string& partition_dir, int effective_rank,
                              int n_effective) {
    // Count total partitions
    int n_partitions = 0;
    for (int r = 0;; ++r) {
        std::string path = partition_dir + "/partition_" + std::to_string(r) + ".h5";
        std::error_code ec;
        if (!std::filesystem::exists(path, ec))
            break;
        ++n_partitions;
    }

    if (n_partitions == 0) {
        throw std::runtime_error("read_partition_range: no partition files found in " +
                                 partition_dir);
    }
    if (effective_rank >= n_effective) {
        throw std::runtime_error(
            "read_partition_range: effective_rank=" + std::to_string(effective_rank) +
            " >= n_effective=" + std::to_string(n_effective));
    }

    // Block distribution
    int base = n_partitions / n_effective;
    int remainder = n_partitions % n_effective;
    int start = effective_rank * base + std::min(effective_rank, remainder);
    int count = base + (effective_rank < remainder ? 1 : 0);

    auto concat_vec = [](auto& dest, const auto& src) {
        dest.insert(dest.end(), src.begin(), src.end());
    };

    RankData merged;
    int cumulative_elements = 0;

    for (int i = 0; i < count; ++i) {
        int part_rank = start + i;
        std::string path = partition_dir + "/partition_" + std::to_string(part_rank) + ".h5";
        RankData part = read_partition(path, part_rank);

        if (i > 0 && part.ngll != merged.ngll) {
            throw std::runtime_error(
                "read_partition_range: partition " + std::to_string(part_rank) +
                " NGLL=" + std::to_string(part.ngll) + " expected " + std::to_string(merged.ngll));
        }

        if (i == 0) {
            merged = std::move(part);
            merged.exchange_patterns.clear();
            merged.ghost_cell_ids.clear();
            merged.ghost_owners.clear();
            cumulative_elements = merged.n_local_cell;
        } else {
            concat_vec(merged.local_cell_ids, part.local_cell_ids);
            concat_vec(merged.coords, part.coords);
            concat_vec(merged.jacobian, part.jacobian);
            concat_vec(merged.dxi_dx, part.dxi_dx);
            concat_vec(merged.mass, part.mass);
            concat_vec(merged.vp, part.vp);
            concat_vec(merged.vs, part.vs);
            concat_vec(merged.density, part.density);
            concat_vec(merged.lambda_, part.lambda_);
            concat_vec(merged.mu_, part.mu_);
            concat_vec(merged.pml_damping, part.pml_damping);

            if (part.recording.has_recording) {
                merged.recording.has_recording = true;
                concat_vec(merged.recording.vertex_ids, part.recording.vertex_ids);
                for (int32_t local_idx : part.recording.src_elem_local) {
                    merged.recording.src_elem_local.push_back(local_idx + cumulative_elements);
                }
                concat_vec(merged.recording.src_corner, part.recording.src_corner);
            }

            cumulative_elements += part.n_local_cell;
        }
    }

    merged.n_local_cell = cumulative_elements;
    merged.n_ghost_cell = 0;
    merged.n_total_cell = merged.n_local_cell;

    // Clear per-rank ibool — cannot merge across partitions
    merged.local_cell2rank_node.clear();
    merged.n_rank_node = 0;

    return merged;
}
ConfigData read_config(const std::string& path) {
    hid_t fid = open_read(path);
    H5FileGuard guard(fid);

    ConfigData cfg;
    cfg.title = "untitled";

    // New config.h5 schema: values are attributes under /simulation.
    hid_t sim_grp = H5Gopen2(fid, "/simulation", H5P_DEFAULT);
    if (sim_grp >= 0) {
        H5FileGuard sim_guard(sim_grp);

        read_attr_string(sim_grp, "title", cfg.title);

        int poly_order = 0;
        if (read_attr_int(sim_grp, "polynomial_order", poly_order) && poly_order > 0) {
            cfg.polynomial_order = poly_order;
        } else {
            cfg.polynomial_order = 3;
        }

        if (!read_attr_double(sim_grp, "solver_dt", cfg.solver_dt)) {
            throw std::runtime_error("Missing required /simulation attribute: solver_dt");
        }
        if (!read_attr_double(sim_grp, "output_dt_s", cfg.output_dt_s)) {
            cfg.output_dt_s = cfg.solver_dt;
        }
        if (!read_attr_int(sim_grp, "snapshot_stride", cfg.snapshot_stride)) {
            cfg.snapshot_stride = 1;
        }
        if (!read_attr_int(sim_grp, "nsteps", cfg.nsteps)) {
            throw std::runtime_error("Missing required /simulation attribute: nsteps");
        }
        read_attr_double(sim_grp, "cfl_safety", cfg.cfl_safety);
        read_attr_string(sim_grp, "snapshot_precision", cfg.snapshot_precision);
        read_attr_double(sim_grp, "record_depth_max_m", cfg.record_depth_max_m);
        read_attr_double(sim_grp, "record_depth_actual_m", cfg.record_depth_actual_m);
        read_attr_int(sim_grp, "nx_elements", cfg.nx_elements);
        read_attr_int(sim_grp, "ny_elements", cfg.ny_elements);
        read_attr_int(sim_grp, "nz_elements", cfg.nz_elements);
        read_attr_int(sim_grp, "pml_xmin", cfg.pml_xmin);
        read_attr_int(sim_grp, "pml_xmax", cfg.pml_xmax);
        read_attr_int(sim_grp, "pml_ymin", cfg.pml_ymin);
        read_attr_int(sim_grp, "pml_ymax", cfg.pml_ymax);
        read_attr_int(sim_grp, "pml_zmin", cfg.pml_zmin);
        read_attr_int(sim_grp, "pml_zmax", cfg.pml_zmax);
        cfg.tilex_elements = read_dataset_int(sim_grp, "tilex_elements");
        cfg.tiley_elements = read_dataset_int(sim_grp, "tiley_elements");
        read_attr_double(sim_grp, "restart_dt_s", cfg.restart_dt_s);
        read_attr_int(sim_grp, "restart_stride", cfg.restart_stride);
        read_attr_int(sim_grp, "log_stride", cfg.log_stride);
    } else {
        // Legacy flat-dataset fallback for old C++ tests/files.
        auto poly_order = try_read_dataset<double>(fid, "polynomial_order");
        cfg.polynomial_order = poly_order.empty() ? 3 : static_cast<int>(poly_order[0]);

        auto dt = try_read_dataset<double>(fid, "dt");
        cfg.solver_dt = dt.empty() ? 0.005 : dt[0];
        cfg.output_dt_s = cfg.solver_dt;

        auto nsteps = try_read_dataset<double>(fid, "nsteps");
        cfg.nsteps = nsteps.empty() ? 1000 : static_cast<int>(nsteps[0]);

        auto cfl = try_read_dataset<double>(fid, "cfl_safety");
        cfg.cfl_safety = cfl.empty() ? 1.0 : cfl[0];

        cfg.snapshot_stride = 1;
        cfg.snapshot_precision = "float64";
        auto use_f32 = try_read_dataset<int64_t>(fid, "use_float32");
        if (!use_f32.empty() && use_f32[0] == 1)
            cfg.snapshot_precision = "float32";
    }

    // Domain bounds: new schema stores attributes under /domain; keep legacy fallback.
    hid_t domain_grp = H5Gopen2(fid, "/domain", H5P_DEFAULT);
    if (domain_grp >= 0) {
        H5FileGuard domain_guard(domain_grp);
        read_attr_double(domain_grp, "xmin", cfg.xmin);
        read_attr_double(domain_grp, "xmax", cfg.xmax);
        read_attr_double(domain_grp, "ymin", cfg.ymin);
        read_attr_double(domain_grp, "ymax", cfg.ymax);
        read_attr_double(domain_grp, "zmin", cfg.zmin);
        read_attr_double(domain_grp, "zmax", cfg.zmax);
    } else {
        auto xmin = try_read_dataset<double>(fid, "xmin");
        auto xmax = try_read_dataset<double>(fid, "xmax");
        auto ymin = try_read_dataset<double>(fid, "ymin");
        auto ymax = try_read_dataset<double>(fid, "ymax");
        auto zmin = try_read_dataset<double>(fid, "zmin");
        auto zmax = try_read_dataset<double>(fid, "zmax");
        if (!xmin.empty())
            cfg.xmin = xmin[0];
        if (!xmax.empty())
            cfg.xmax = xmax[0];
        if (!ymin.empty())
            cfg.ymin = ymin[0];
        if (!ymax.empty())
            cfg.ymax = ymax[0];
        if (!zmin.empty())
            cfg.zmin = zmin[0];
        if (!zmax.empty())
            cfg.zmax = zmax[0];
    }

    // Source data: new schema stores datasets/attrs under /source; keep legacy fallback.
    hid_t source_grp = H5Gopen2(fid, "/source", H5P_DEFAULT);
    if (source_grp >= 0) {
        H5FileGuard source_guard(source_grp);
        cfg.stf_t = try_read_dataset<double>(fid, "/source/stf_t");
        cfg.stf_values = try_read_dataset<double>(fid, "/source/stf_values");
        read_attr_double(source_grp, "x", cfg.source_x);
        read_attr_double(source_grp, "y", cfg.source_y);
        read_attr_double(source_grp, "z", cfg.source_z);
    } else {
        cfg.stf_t = try_read_dataset<double>(fid, "stf_t");
        cfg.stf_values = try_read_dataset<double>(fid, "stf_values");
        auto src_x = try_read_dataset<double>(fid, "source_x");
        auto src_y = try_read_dataset<double>(fid, "source_y");
        auto src_z = try_read_dataset<double>(fid, "source_z");
        if (!src_x.empty())
            cfg.source_x = src_x[0];
        if (!src_y.empty())
            cfg.source_y = src_y[0];
        if (!src_z.empty())
            cfg.source_z = src_z[0];
    }

    // Source element location data (precomputed by source_locator.py)
    cfg.src_cell_ids = try_read_dataset<int64_t>(fid, "/source/elements/element_ids");
    cfg.src_weights = try_read_dataset<double>(fid, "/source/elements/weights");
    cfg.n_src_cell = static_cast<int>(cfg.src_cell_ids.size());
    return cfg;
}

}  // namespace gf