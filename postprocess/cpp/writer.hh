#ifndef GF_POST_WRITER_HH
#define GF_POST_WRITER_HH

/* writer.hh — HDF5 tile writer for Green's function output.
 *
 * Writes tile_x{i}_y{j}.h5 files with vertex_ids + greens_tensor.
 * Supports element-count tiling and spatial tiling (green_tile_size_m).
 * Output format matches Python gf_post.writer.GFWriter.
 */

#include <hdf5.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "reader.hh"  // for ConfigParams, ModelData

// -----------------------------------------------------------------------
// Tile index helpers
// -----------------------------------------------------------------------

// Find tile index given interior element index and tile_sizes array.
inline int find_tile_index(int64_t interior_idx, const std::vector<int64_t>& tile_sizes) {
    int64_t cum = 0;
    for (size_t t = 0; t < tile_sizes.size(); ++t) {
        cum += tile_sizes[t];
        if (interior_idx < cum)
            return (int)t;
    }
    return (int)(tile_sizes.size()) - 1;
}

// -----------------------------------------------------------------------
// Tile binning: returns map from (tx, ty) → list of vertex indices
// -----------------------------------------------------------------------

struct TileKey {
    int tx, ty;
    bool operator<(const TileKey& o) const { return (tx < o.tx) || (tx == o.tx && ty < o.ty); }
    bool operator==(const TileKey& o) const { return tx == o.tx && ty == o.ty; }
};

// Hash for TileKey
struct TileKeyHash {
    size_t operator()(const TileKey& k) const { return ((size_t)k.tx << 16) ^ (size_t)k.ty; }
};

#include <unordered_map>

struct TileBins {
    // For each tile key, the list of vertex indices (0-based global)
    std::unordered_map<TileKey, std::vector<int64_t>, TileKeyHash> bins;
    // Sorted tile keys
    std::vector<TileKey> keys;
};

// Forward declarations for attr helpers (defined after write_tile)
inline void write_double_attr_into(hid_t loc, const char* name, double val);
inline void write_int_attr_into(hid_t loc, const char* name, int val);
// -----------------------------------------------------------------------
// Write a tensor dataset with configurable precision. dims=[nt, n_local, ...].
// use_float32=true writes float32 (with conversion), false writes double directly.
inline void write_tensor_ds(hid_t gid, const char* name, const double* data, const hsize_t* dims,
                            int ndims, bool use_float32) {
    hid_t space = H5Screate_simple(ndims, dims, nullptr);
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_shuffle(plist);
    hsize_t chunk[4];
    chunk[0] = 1;
    for (int d = 1; d < ndims; ++d)
        chunk[d] = dims[d];
    if (ndims > 1 && dims[1] == 0)
        chunk[1] = 1;
    H5Pset_chunk(plist, ndims, chunk);
    H5Pset_deflate(plist, 4);

    if (use_float32) {
        size_t total = 1;
        for (int d = 0; d < ndims; ++d)
            total *= (size_t)dims[d];
        std::vector<float> fbuf(total);
        for (size_t i = 0; i < total; ++i)
            fbuf[i] = static_cast<float>(data[i]);
        hid_t ds = H5Dcreate2(gid, name, H5T_NATIVE_FLOAT, space, H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fbuf.data());
        H5Dclose(ds);
    } else {
        hid_t ds =
            H5Dcreate2(gid, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, plist, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5Dclose(ds);
    }
    H5Pclose(plist);
    H5Sclose(space);
}

// Write a single tile HDF5 file
// -----------------------------------------------------------------------

inline void write_tile(
    const std::string& path, int tile_x, int tile_y, double x_min_m, double x_max_m,
    double y_min_m, double y_max_m, double z_min_m, double z_max_m, double record_depth_max_m,
    double record_depth_actual_m,
    const std::vector<int64_t>& tile_vertex_ids,  // 1-based
    const std::vector<double>& time_arr,          // [nt]
    double solver_dt_s,
    const std::vector<double>& tile_greens,  // [nt, n_local, 6, 3]
    const double source_xyz_m[3],
    const std::vector<double>& tile_vertex_coords,  // [n_local, 3]
    const double* displacement_tensor,              // nullptr = strain-only
    const double* velocity_tensor = nullptr,        // [nt, n_local, 3, 3]
    const double* acceleration_tensor = nullptr,    // [nt, n_local, 3, 3]
    const std::vector<double>& stf_t = {},  // [nt] STF time [s], downsampled to output_dt_s
    const std::vector<double>& stf_values =
        {},  // [nt] STF amplitude [N], downsampled to output_dt_s
    bool use_float32 = true) {
    hid_t fid = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "ERROR: cannot create %s\n", path.c_str());
        return;
    }

    int64_t nt = (int64_t)time_arr.size();
    int64_t n_local = (int64_t)tile_vertex_ids.size();
    int64_t ncomp = 6;
    int64_t ndir = 3;

    // ---- Attrs ----
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, 64);

    auto write_str_attr = [&](const char* name, const char* val) {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(fid, name, str_type, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, str_type, val);
        H5Aclose(attr);
        H5Sclose(space);
    };
    auto write_double_attr = [&](const char* name, double val) {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(fid, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_DOUBLE, &val);
        H5Aclose(attr);
        H5Sclose(space);
    };
    auto write_int_attr = [&](const char* name, int val) {
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(fid, name, H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT32, &val);
        H5Aclose(attr);
        H5Sclose(space);
    };

    write_str_attr("version", "1.0.0");
    write_str_attr("basis", "gll");
    write_int_attr("tile_x_index", tile_x);
    write_int_attr("tile_y_index", tile_y);
    write_double_attr("x_min_m", x_min_m);
    write_double_attr("x_max_m", x_max_m);
    write_double_attr("y_min_m", y_min_m);
    write_double_attr("y_max_m", y_max_m);
    write_double_attr("z_min_m", z_min_m);
    write_double_attr("z_max_m", z_max_m);
    write_double_attr("record_depth_max_m", record_depth_max_m);
    write_double_attr("record_depth_actual_m", record_depth_actual_m);

    // Source position attrs
    {
        hsize_t sdims[1] = {3};
        hid_t space = H5Screate_simple(1, sdims, nullptr);
        hid_t attr =
            H5Acreate2(fid, "source_xyz_m", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_DOUBLE, source_xyz_m);
        H5Aclose(attr);
        H5Sclose(space);
    }
    write_str_attr("source_directions", "x,y,z");
    std::string qstr = "strain";
    if (displacement_tensor)
        qstr += ",displacement";
    if (velocity_tensor)
        qstr += ",velocity";
    if (acceleration_tensor)
        qstr += ",acceleration";
    write_str_attr("greens_quantities", qstr.c_str());

    // excludes_pml: int
    {
        int true_val = 1;
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr =
            H5Acreate2(fid, "excludes_pml", H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT32, &true_val);
        H5Aclose(attr);
        H5Sclose(space);
    }

    // ---- /time group ----
    hid_t time_gid = H5Gcreate2(fid, "time", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    {
        hsize_t tdim = (hsize_t)nt;
        hid_t space = H5Screate_simple(1, &tdim, nullptr);
        hid_t ds = H5Dcreate2(time_gid, "t", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, time_arr.data());
        H5Dclose(ds);
        H5Sclose(space);

        // attrs
        write_double_attr_into(time_gid, "dt", solver_dt_s);
        write_int_attr_into(time_gid, "nsteps", (int)nt);
    }
    H5Gclose(time_gid);

    // ---- /mesh group ----
    hid_t mesh_gid = H5Gcreate2(fid, "mesh", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    {
        hsize_t vdim = (hsize_t)n_local;
        hid_t space = H5Screate_simple(1, &vdim, nullptr);
        hid_t ds = H5Dcreate2(mesh_gid, "gll_node_ids", H5T_NATIVE_INT64, space, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT);
        // Convert to int64 if needed
        std::vector<int64_t> ids(n_local);
        for (int64_t i = 0; i < n_local; ++i)
            ids[i] = tile_vertex_ids[i];
        H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, ids.data());
        H5Dclose(ds);
        H5Sclose(space);
    }
    // vertex_coords: [n_local, 3] float64
    if (!tile_vertex_coords.empty()) {
        hsize_t cdims[2] = {(hsize_t)n_local, 3};
        hid_t space = H5Screate_simple(2, cdims, nullptr);
        hid_t ds = H5Dcreate2(mesh_gid, "gll_node_coords", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tile_vertex_coords.data());
        H5Dclose(ds);
        H5Sclose(space);
    }
    H5Gclose(mesh_gid);

    // ---- /source group (STF time series) ----
    // Stored as float64 (metadata, not compressed). The STF convolved with
    // the Green's tensor produces the recorded response; storing it lets
    // users deconvolve to recover the impulse response if desired.
    if (!stf_t.empty()) {
        hid_t src_gid = H5Gcreate2(fid, "source", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t sdim = (hsize_t)stf_t.size();
        hid_t space = H5Screate_simple(1, &sdim, nullptr);
        hid_t ds_t = H5Dcreate2(src_gid, "stf_t", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds_t, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, stf_t.data());
        H5Dclose(ds_t);
        hid_t ds_v = H5Dcreate2(src_gid, "stf_values", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                                H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds_v, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, stf_values.data());
        H5Dclose(ds_v);
        H5Sclose(space);
        H5Gclose(src_gid);
    }

    // ---- /field group ----
    hid_t field_gid = H5Gcreate2(fid, "field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    {
        // greens_tensor: [nt, n_local, 6, 3]
        hsize_t gdims[4] = {(hsize_t)nt, (hsize_t)n_local, (hsize_t)ncomp, (hsize_t)ndir};
        write_tensor_ds(field_gid, "greens_tensor", tile_greens.data(), gdims, 4, use_float32);

        // displacement_tensor: [nt, n_local, 3, 3] (optional)
        if (displacement_tensor != nullptr) {
            hsize_t ddims[4] = {(hsize_t)nt, (hsize_t)n_local, 3, 3};
            write_tensor_ds(field_gid, "displacement_tensor", displacement_tensor, ddims, 4,
                            use_float32);
        }

        // velocity_tensor: [nt, n_local, 3, 3] (optional)
        if (velocity_tensor != nullptr) {
            hsize_t vdims[4] = {(hsize_t)nt, (hsize_t)n_local, 3, 3};
            write_tensor_ds(field_gid, "velocity_tensor", velocity_tensor, vdims, 4, use_float32);
        }

        // acceleration_tensor: [nt, n_local, 3, 3] (optional)
        if (acceleration_tensor != nullptr) {
            hsize_t adims[4] = {(hsize_t)nt, (hsize_t)n_local, 3, 3};
            write_tensor_ds(field_gid, "acceleration_tensor", acceleration_tensor, adims, 4,
                            use_float32);
        }
    }
    H5Gclose(field_gid);

    H5Tclose(str_type);
    H5Fclose(fid);
}

// Helper: write double attr into an open group
inline void write_double_attr_into(hid_t loc, const char* name, double val) {
    if (H5Aexists(loc, name))
        H5Adelete(loc, name);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
    H5Sclose(space);
}

// Helper: write int attr into an open group
inline void write_int_attr_into(hid_t loc, const char* name, int val) {
    if (H5Aexists(loc, name))
        H5Adelete(loc, name);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT32, &val);
    H5Aclose(attr);
    H5Sclose(space);
}

#endif  // GF_POST_WRITER_HH