#ifndef GF_POST_READER_HH
#define GF_POST_READER_HH

/* reader.hh — HDF5 readers for postprocess
 *
 * ConfigReader : /simulation/ attrs + tile arrays from config.h5
 * ModelReader  : /topology/vertex_to_coord + /domain/ bounds from model.h5
 * RecordScanner: discover record_{r}_{step}.h5 files in a directory
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

// -----------------------------------------------------------------------
// HDF5 helpers (same pattern as preprocess/cpp/)
// -----------------------------------------------------------------------

static hid_t open_or_fail(const char* path, unsigned flags) {
    hid_t fid = H5Fopen(path, flags, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }
    return fid;
}

static void read_attr_double(hid_t loc, const char* name, double& val) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) {
        val = 0.0;
        return;
    }
    H5Aread(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
}

static void read_attr_int64(hid_t loc, const char* name, int64_t& val) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) {
        val = 0;
        return;
    }
    H5Aread(attr, H5T_NATIVE_INT64, &val);
    H5Aclose(attr);
}

// Read a string attribute (fixed-size or variable-length)
static std::string read_attr_string(hid_t loc, const char* name,
                                    const std::string& fallback = "") {
    if (H5Aexists(loc, name) <= 0)
        return fallback;
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0)
        return fallback;
    hid_t ftype = H5Aget_type(attr);
    if (ftype < 0) {
        H5Aclose(attr);
        return fallback;
    }
    std::string result = fallback;
    if (H5Tget_class(ftype) == H5T_STRING) {
        if (H5Tis_variable_str(ftype) > 0) {
            char* value = nullptr;
            if (H5Aread(attr, ftype, &value) >= 0) {
                result = value ? std::string(value) : std::string();
                if (value)
                    H5free_memory(value);
            }
        } else {
            size_t sz = (size_t)H5Tget_size(ftype);
            std::vector<char> buf(sz + 1, '\0');
            if (H5Aread(attr, ftype, buf.data()) >= 0)
                result = std::string(buf.data());
        }
    }
    H5Tclose(ftype);
    H5Aclose(attr);
    return result;
}

// Read 1-D int64 dataset
static std::vector<int64_t> read_int64_1d(hid_t loc, const char* name, hsize_t& n) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) {
        n = 0;
        return {};
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n = dims[0];
    std::vector<int64_t> buf(n);
    H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

// Read 1-D double dataset
static std::vector<double> read_double_1d(hid_t loc, const char* name, hsize_t& n) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) {
        n = 0;
        return {};
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n = dims[0];
    std::vector<double> buf(n);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

// Read 2-D double array [n, 3]
static std::vector<double> read_vertex_coords(hid_t loc) {
    hid_t ds = H5Dopen2(loc, "vertex_to_coord", H5P_DEFAULT);
    if (ds < 0) {
        fprintf(stderr, "ERROR: vertex_to_coord not found\n");
        exit(1);
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    hsize_t nv = dims[0];
    std::vector<double> buf(nv * 3);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

// Read a 3-D float32 array [1, n_vertices, 6] (strain snapshot)
static void read_strain_1xNx6(hid_t loc, const char* name, hsize_t& n_vertices,
                              std::vector<double>& buf) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) {
        n_vertices = 0;
        return;
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[3];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n_vertices = dims[1];
    hsize_t total = dims[0] * dims[1] * dims[2];
    buf.resize(total);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
}

// Read a 3-D float32 array [1, n_vertices, 3] (displacement/velocity/acceleration snapshot)
static void read_field_1xNx3(hid_t loc, const char* name, hsize_t& n_vertices,
                             std::vector<double>& buf) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) {
        n_vertices = 0;
        return;
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[3];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n_vertices = dims[1];
    hsize_t total = dims[0] * dims[1] * dims[2];
    buf.resize(total);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
}

// ----- GLL 4D record format (post global-DOF repair) -----

// Read a 4-D array [1, n_rec_cell, n_node_per_cell, ncomp] (strain snapshot)
static void read_strain_4d(hid_t loc, const char* name, hsize_t& n_rec_cell,
                           hsize_t& n_node_per_cell, std::vector<double>& buf) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) {
        n_rec_cell = 0;
        n_node_per_cell = 0;
        return;
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[4];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n_rec_cell = dims[1];
    n_node_per_cell = dims[2];
    hsize_t total = dims[0] * dims[1] * dims[2] * dims[3];
    buf.resize(total);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
}

// Read a 4-D array [1, n_rec_cell, n_node_per_cell, 3] (displacement)
static void read_field_4d(hid_t loc, const char* name, hsize_t& n_rec_cell,
                          hsize_t& n_node_per_cell, std::vector<double>& buf) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) {
        n_rec_cell = 0;
        n_node_per_cell = 0;
        return;
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[4];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n_rec_cell = dims[1];
    n_node_per_cell = dims[2];
    hsize_t total = dims[0] * dims[1] * dims[2] * dims[3];
    buf.resize(total);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
}

// -----------------------------------------------------------------------
// ConfigReader
// -----------------------------------------------------------------------

struct ConfigParams {
    int64_t nx_elements = 0;
    int64_t ny_elements = 0;
    double solver_dt = 0.01;
    double output_dt_s = 0.01;
    int64_t nsteps = 0;
    double record_depth_max_m = 0.0;
    double record_depth_actual_m = 0.0;
    std::string snapshot_precision = "float64";  // "float32" or "float64"
    // Source position
    double source_x_m = 0.0;
    double source_y_m = 0.0;
    double source_z_m = 0.0;
    // PML thickness
    int64_t pml_xmin = 0, pml_xmax = 0;
    int64_t pml_ymin = 0, pml_ymax = 0;
    int64_t pml_zmin = 0, pml_zmax = 0;
    // Source time function (from /source/stf_t, /source/stf_values)
    std::vector<double> stf_t;       // [nsteps] time points [s]
    std::vector<double> stf_values;  // [nsteps] force amplitude [N]
    // Tile arrays
    std::vector<int64_t> tilex_elements;
    std::vector<int64_t> tiley_elements;
};

inline ConfigParams read_config(const char* config_path) {
    ConfigParams cfg;
    hid_t fid = open_or_fail(config_path, H5F_ACC_RDONLY);
    hid_t sim_gid = H5Gopen2(fid, "simulation", H5P_DEFAULT);
    if (sim_gid < 0) {
        fprintf(stderr, "ERROR: /simulation group not found in %s\n", config_path);
        H5Fclose(fid);
        exit(1);
    }

    read_attr_int64(sim_gid, "nx_elements", cfg.nx_elements);
    read_attr_int64(sim_gid, "ny_elements", cfg.ny_elements);
    read_attr_double(sim_gid, "solver_dt", cfg.solver_dt);
    read_attr_double(sim_gid, "output_dt_s", cfg.output_dt_s);
    read_attr_int64(sim_gid, "nsteps", cfg.nsteps);
    read_attr_double(sim_gid, "record_depth_max_m", cfg.record_depth_max_m);
    read_attr_double(sim_gid, "record_depth_actual_m", cfg.record_depth_actual_m);
    cfg.snapshot_precision = read_attr_string(sim_gid, "snapshot_precision", "float64");

    // PML thickness
    read_attr_int64(sim_gid, "pml_xmin", cfg.pml_xmin);
    read_attr_int64(sim_gid, "pml_xmax", cfg.pml_xmax);
    read_attr_int64(sim_gid, "pml_ymin", cfg.pml_ymin);
    read_attr_int64(sim_gid, "pml_ymax", cfg.pml_ymax);
    read_attr_int64(sim_gid, "pml_zmin", cfg.pml_zmin);
    read_attr_int64(sim_gid, "pml_zmax", cfg.pml_zmax);

    // Tile arrays (datasets)
    hsize_t n;
    auto tx = read_int64_1d(sim_gid, "tilex_elements", n);
    cfg.tilex_elements = std::move(tx);
    auto ty = read_int64_1d(sim_gid, "tiley_elements", n);
    cfg.tiley_elements = std::move(ty);

    // Read /source group attrs
    hid_t src_gid = H5Gopen2(fid, "source", H5P_DEFAULT);
    if (src_gid >= 0) {
        read_attr_double(src_gid, "x", cfg.source_x_m);
        read_attr_double(src_gid, "y", cfg.source_y_m);
        read_attr_double(src_gid, "z", cfg.source_z_m);
        // Read STF time series [nsteps] each
        hsize_t nstf = 0;
        cfg.stf_t = read_double_1d(src_gid, "stf_t", nstf);
        cfg.stf_values = read_double_1d(src_gid, "stf_values", nstf);
        H5Gclose(src_gid);
    }

    H5Gclose(sim_gid);
    H5Fclose(fid);

    // If output_dt_s wasn't explicitly set, use solver_dt
    if (cfg.output_dt_s <= 0)
        cfg.output_dt_s = cfg.solver_dt;

    return cfg;
}

// -----------------------------------------------------------------------
// ModelReader
// -----------------------------------------------------------------------

struct ModelData {
    int64_t n_vertex = 0;
    std::vector<double> vertex_coords;  // [n_vertex, 3]
    double xmin = 0, xmax = 0;
    double ymin = 0, ymax = 0;
    double zmin = 0, zmax = 0;
};

inline ModelData read_model(const char* model_path) {
    ModelData md;
    hid_t fid = open_or_fail(model_path, H5F_ACC_RDONLY);

    // Topology
    hid_t topo_gid = H5Gopen2(fid, "topology", H5P_DEFAULT);
    if (topo_gid < 0) {
        fprintf(stderr, "ERROR: /topology group not found in %s\n", model_path);
        H5Fclose(fid);
        exit(1);
    }
    md.vertex_coords = read_vertex_coords(topo_gid);
    md.n_vertex = (int64_t)(md.vertex_coords.size() / 3);
    H5Gclose(topo_gid);

    // Domain bounds
    hid_t dom_gid = H5Gopen2(fid, "domain", H5P_DEFAULT);
    if (dom_gid >= 0) {
        read_attr_double(dom_gid, "xmin", md.xmin);
        read_attr_double(dom_gid, "xmax", md.xmax);
        read_attr_double(dom_gid, "ymin", md.ymin);
        read_attr_double(dom_gid, "ymax", md.ymax);
        read_attr_double(dom_gid, "zmin", md.zmin);
        read_attr_double(dom_gid, "zmax", md.zmax);
        H5Gclose(dom_gid);
    } else {
        // Infer from vertex coords
        if (md.n_vertex > 0) {
            md.xmin = md.ymin = md.zmin = 1e30;
            md.xmax = md.ymax = md.zmax = -1e30;
            for (int64_t i = 0; i < md.n_vertex; ++i) {
                double x = md.vertex_coords[i * 3 + 0];
                double y = md.vertex_coords[i * 3 + 1];
                double z = md.vertex_coords[i * 3 + 2];
                if (x < md.xmin)
                    md.xmin = x;
                if (x > md.xmax)
                    md.xmax = x;
                if (y < md.ymin)
                    md.ymin = y;
                if (y > md.ymax)
                    md.ymax = y;
                if (z < md.zmin)
                    md.zmin = z;
                if (z > md.zmax)
                    md.zmax = z;
            }
        }
    }

    H5Fclose(fid);
    return md;
}

// -----------------------------------------------------------------------
// Record file discovery & reading
// -----------------------------------------------------------------------

struct RecordFileInfo {
    std::string path;
    int rank = -1;
    int step = -1;
};

// Parse filename: record_{rank}_{step}.h5 or legacy record_{rank}.h5
inline bool parse_record_filename(const std::string& basename, int& rank, int& step,
                                  bool& legacy) {
    legacy = false;
    int n = std::sscanf(basename.c_str(), "record_%d_%d.h5", &rank, &step);
    if (n == 2)
        return true;
    n = std::sscanf(basename.c_str(), "record_%d.h5", &rank);
    if (n == 1) {
        legacy = true;
        step = 0;
        return true;
    }
    return false;
}

// Discover record files via system glob (POSIX)
#include <glob.h>

inline std::vector<RecordFileInfo> discover_records(const char* dir_path) {
    std::string pattern = std::string(dir_path) + "/record_*.h5";
    glob_t gl;
    int ret = glob(pattern.c_str(), GLOB_NOSORT, nullptr, &gl);
    if (ret != 0) {
        return {};
    }

    std::vector<RecordFileInfo> files;
    for (size_t i = 0; i < gl.gl_pathc; ++i) {
        std::string fpath(gl.gl_pathv[i]);
        std::string basename = fpath.substr(fpath.find_last_of('/') + 1);
        RecordFileInfo info;
        info.path = fpath;
        bool legacy = false;
        if (parse_record_filename(basename, info.rank, info.step, legacy)) {
            files.push_back(info);
        }
    }
    globfree(&gl);
    return files;
}

// Group record files by step, return sorted step list and per-step file lists
struct StepGroup {
    int step;
    std::vector<RecordFileInfo> files;
};

inline std::vector<StepGroup> group_by_step(const std::vector<RecordFileInfo>& files) {
    // Find unique steps
    std::vector<int> steps;
    for (auto& f : files) {
        if (std::find(steps.begin(), steps.end(), f.step) == steps.end())
            steps.push_back(f.step);
    }
    std::sort(steps.begin(), steps.end());

    std::vector<StepGroup> groups;
    for (int s : steps) {
        StepGroup g;
        g.step = s;
        for (auto& f : files)
            if (f.step == s)
                g.files.push_back(f);
        groups.push_back(g);
    }
    return groups;
}

// Read a single record file and scatter strain by vertex_id into full array.
// full_strain: [n_vertex, 6] output (pre-allocated, zero-initialized)
// full_mask: [n_vertex] bool output (pre-allocated, false-initialized)
// Returns false on error.
inline bool read_record_into(const RecordFileInfo& fi, int64_t n_vertex,
                             std::vector<double>& full_strain,  // [n_vertex, 6]
                             std::vector<bool>& full_mask,
                             std::vector<double>& full_displacement,  // [n_vertex, 3]
                             std::vector<double>& full_velocity,      // [n_vertex, 3]
                             std::vector<double>& full_acceleration   // [n_vertex, 3]
) {
    hid_t fid = H5Fopen(fi.path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "WARNING: cannot open %s\n", fi.path.c_str());
        return false;
    }

    // Read vertex_ids
    hsize_t n_local = 0;
    auto local_ids = read_int64_1d(fid, "vertex_ids", n_local);
    if (n_local == 0) {
        H5Fclose(fid);
        return true;
    }

    // Read strain [1, n_local, 6]
    hsize_t nv = 0;
    std::vector<double> strain_buf;
    read_strain_1xNx6(fid, "strain", nv, strain_buf);
    if (nv != n_local) {
        fprintf(stderr, "WARNING: vertex_ids/strain size mismatch in %s\n", fi.path.c_str());
        H5Fclose(fid);
        return false;
    }

    // Read displacement [1, n_local, 3] (optional — backward compat)
    hsize_t ndv = 0;
    std::vector<double> disp_buf;
    read_field_1xNx3(fid, "displacement", ndv, disp_buf);
    bool has_displacement = (ndv == n_local && !disp_buf.empty());

    // Read velocity [1, n_local, 3] (optional)
    hsize_t nvv = 0;
    std::vector<double> vel_buf;
    read_field_1xNx3(fid, "velocity", nvv, vel_buf);
    bool has_velocity = (nvv == n_local && !vel_buf.empty());

    // Read acceleration [1, n_local, 3] (optional)
    hsize_t nav = 0;
    std::vector<double> acc_buf;
    read_field_1xNx3(fid, "acceleration", nav, acc_buf);
    bool has_acceleration = (nav == n_local && !acc_buf.empty());

    H5Fclose(fid);

    // Scatter strain by vertex_id (1-based → 0-based)
    for (hsize_t li = 0; li < n_local; ++li) {
        int64_t gid = local_ids[li] - 1;  // 1-based → 0-based
        if (gid < 0 || gid >= n_vertex) {
            fprintf(stderr, "WARNING: vertex_id %lld out of range [0, %lld) in %s\n",
                    (long long)(gid + 1), (long long)n_vertex, fi.path.c_str());
            continue;
        }
        // Partition-boundary vertices may appear in multiple rank files.
        // The last rank's value is used, which is correct for non-overlapping
        // partition ownership.
        double* src = strain_buf.data() + li * 6;
        double* dst = full_strain.data() + gid * 6;
        for (int c = 0; c < 6; ++c)
            dst[c] = src[c];
        full_mask[gid] = true;

        // Scatter displacement if available
        if (has_displacement) {
            double* dsrc = disp_buf.data() + li * 3;
            double* ddst = full_displacement.data() + gid * 3;
            for (int c = 0; c < 3; ++c)
                ddst[c] = dsrc[c];
        }

        // Scatter velocity if available
        if (has_velocity) {
            double* vsrc = vel_buf.data() + li * 3;
            double* vdst = full_velocity.data() + gid * 3;
            for (int c = 0; c < 3; ++c)
                vdst[c] = vsrc[c];
        }

        // Scatter acceleration if available
        if (has_acceleration) {
            double* asrc = acc_buf.data() + li * 3;
            double* adst = full_acceleration.data() + gid * 3;
            for (int c = 0; c < 3; ++c)
                adst[c] = asrc[c];
        }
    }
    return true;
}

#endif  // GF_POST_READER_HH