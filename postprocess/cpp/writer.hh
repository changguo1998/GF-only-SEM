#ifndef GF_POST_WRITER_HH
#define GF_POST_WRITER_HH

/* writer.hh — HDF5 tile writer for Green's function output.
 *
 * Writes tile_x{i}_y{j}.h5 files with vertex_ids + greens_tensor.
 * Supports element-count tiling and spatial tiling (green_tile_size_m).
 * Output format matches Python gf_post.writer.GFWriter.
 */

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

inline TileBins bin_vertices(
    const ConfigParams& cfg, const ModelData& model,
    const std::vector<int64_t>& recorded_vertex_ids,  // 1-based, subset of all vertices
    int64_t n_recorded) {
    TileBins result;
    bool use_spatial = (cfg.green_tile_size_m > 0);

    double xmin = model.xmin, ymin = model.ymin;
    double xmax = model.xmax, ymax = model.ymax;

    for (int64_t vi = 0; vi < n_recorded; ++vi) {
        int64_t gid = recorded_vertex_ids[vi] - 1;  // 0-based
        double x = model.vertex_coords[gid * 3 + 0];
        double y = model.vertex_coords[gid * 3 + 1];

        TileKey key;
        if (use_spatial) {
            double gts = cfg.green_tile_size_m;
            key.tx = (int)std::floor((x - xmin) / gts);
            key.ty = (int)std::floor((y - ymin) / gts);
        } else {
            double dx = (xmax - xmin) / cfg.nx_elements;
            double dy = (ymax - ymin) / cfg.ny_elements;
            int64_t ei = (dx > 0) ? (int64_t)std::floor((x - xmin) / dx) : 0;
            int64_t ej = (dy > 0) ? (int64_t)std::floor((y - ymin) / dy) : 0;
            if (ei < 0)
                ei = 0;
            if (ej < 0)
                ej = 0;
            if (cfg.nx_elements > 0 && ei >= cfg.nx_elements)
                ei = cfg.nx_elements - 1;
            if (cfg.ny_elements > 0 && ej >= cfg.ny_elements)
                ej = cfg.ny_elements - 1;

            int64_t interior_i = ei - cfg.pml_xmin;
            int64_t interior_j = ej - cfg.pml_ymin;

            // Compute total interior elements from tile arrays
            int64_t total_interior_x = 0;
            for (auto sz : cfg.tilex_elements)
                total_interior_x += sz;
            int64_t total_interior_y = 0;
            for (auto sz : cfg.tiley_elements)
                total_interior_y += sz;

            if (interior_i < 0 || interior_i >= total_interior_x)
                continue;
            if (interior_j < 0 || interior_j >= total_interior_y)
                continue;

            key.tx = find_tile_index(interior_i, cfg.tilex_elements);
            key.ty = find_tile_index(interior_j, cfg.tiley_elements);
        }

        result.bins[key].push_back(vi);
    }

    // Sort keys
    for (auto& kv : result.bins) {
        result.keys.push_back(kv.first);
    }
    std::sort(result.keys.begin(), result.keys.end());

    return result;
}

// Forward declarations for attr helpers (defined after write_tile)
inline void write_double_attr_into(hid_t loc, const char* name, double val);
inline void write_int_attr_into(hid_t loc, const char* name, int val);
// -----------------------------------------------------------------------
// Write a single tile HDF5 file
// -----------------------------------------------------------------------

inline void write_tile(const std::string& path, int tile_x, int tile_y, double x_min_m,
                       double x_max_m, double y_min_m, double y_max_m, double z_min_m,
                       double z_max_m, double record_depth_max_m, double record_depth_actual_m,
                       const std::vector<int64_t>& tile_vertex_ids,  // 1-based
                       const std::vector<double>& time_arr,          // [nt]
                       double solver_dt_s,
                       const std::vector<float>& tile_greens,  // [nt, n_local, 6, 3]
                       const double source_xyz_m[3],
                       const std::vector<double>& tile_vertex_coords,  // [n_local, 3]
                       const float* displacement_tensor  // nullptr = strain-only
) {
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
    write_str_attr("basis", "mesh_vertices");
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
        hid_t attr = H5Acreate2(fid, "source_xyz_m", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_DOUBLE, source_xyz_m);
        H5Aclose(attr);
        H5Sclose(space);
    }
    write_str_attr("source_directions", "x,y,z");
    write_str_attr("greens_quantities", displacement_tensor ? "strain,displacement" : "strain");

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
        hid_t ds = H5Dcreate2(mesh_gid, "vertex_ids", H5T_NATIVE_INT64, space, H5P_DEFAULT,
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
        hid_t ds = H5Dcreate2(mesh_gid, "vertex_coords", H5T_NATIVE_DOUBLE, space, H5P_DEFAULT,
                              H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tile_vertex_coords.data());
        H5Dclose(ds);
        H5Sclose(space);
    }
    H5Gclose(mesh_gid);

    // ---- /field group ----
    hid_t field_gid = H5Gcreate2(fid, "field", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    {
        // greens_tensor: [nt, n_local, 6, 3] float32
        hsize_t gdims[4] = {(hsize_t)nt, (hsize_t)n_local, (hsize_t)ncomp, (hsize_t)ndir};
        hid_t space = H5Screate_simple(4, gdims, nullptr);

        // Enable gzip + shuffle compression
        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_shuffle(plist);
        hsize_t chunk_dims[4] = {1, (hsize_t)n_local, (hsize_t)ncomp, (hsize_t)ndir};
        if (n_local == 0)
            chunk_dims[1] = 1;
        H5Pset_chunk(plist, 4, chunk_dims);
        H5Pset_deflate(plist, 4);

        hid_t ds = H5Dcreate2(field_gid, "greens_tensor", H5T_NATIVE_FLOAT, space, H5P_DEFAULT,
                              plist, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, tile_greens.data());
        H5Dclose(ds);
        H5Pclose(plist);
        H5Sclose(space);

        // displacement_tensor: [nt, n_local, 3, 3] float32 (optional)
        if (displacement_tensor != nullptr) {
            hsize_t ddims[4] = {(hsize_t)nt, (hsize_t)n_local, 3, 3};
            hid_t dspace = H5Screate_simple(4, ddims, nullptr);
            hid_t dplist = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_shuffle(dplist);
            hsize_t dchunk[4] = {1, (hsize_t)n_local, 3, 3};
            if (n_local == 0)
                dchunk[1] = 1;
            H5Pset_chunk(dplist, 4, dchunk);
            H5Pset_deflate(dplist, 4);
            hid_t dds = H5Dcreate2(field_gid, "displacement_tensor", H5T_NATIVE_FLOAT, dspace,
                                   H5P_DEFAULT, dplist, H5P_DEFAULT);
            H5Dwrite(dds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, displacement_tensor);
            H5Dclose(dds);
            H5Pclose(dplist);
            H5Sclose(dspace);
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