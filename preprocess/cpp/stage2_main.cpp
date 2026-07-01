/* preprocess/cpp/stage2_main.cpp — Second stage: λ/μ, solver_dt, pre-flight.
 *
 * Standalone executable (no MPI).  Reads model.h5 with topology, GLL geometry,
 * material arrays (vp/vs/density), and /config/ attrs written by Python.
 * Computes λ, μ, solver timestep, nsteps, and pre-flight statistics.
 *
 * Usage:
 *   gf_preprocess_stage2 <model.h5>
 *
 * Reads from model.h5:
 *   /topology/                     — n_cell, connectivity
 *   /field/element/{coords,jacobian,vp,vs,density}
 *   /field/surface/boundary_tag
 *   /config/{cfl_safety,output_dt_s,total_duration_s,n_ranks,
 *            snapshot_precision,storage_limit_gb,record_depth_max_m,
 *            nx_elements,ny_elements,NGLL}
 *
 * Writes to model.h5:
 *   /field/element/{lambda,mu}
 *
 * Prints to stdout (machine-parseable for Python):
 *   STAT_<key>=<value>
 */

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static const int MAX_STRIDE = 100;

// -----------------------------------------------------------------------
// HDF5 helpers
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

static void read_attr_str(hid_t loc, const char* name, char* buf, size_t bufsz) {
    buf[0] = 0;
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0)
        return;
    hid_t atype = H5Aget_type(attr);
    // If variable-length string, read as fixed-size
    hid_t mem_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(mem_type, bufsz);
    H5Aread(attr, mem_type, buf);
    buf[bufsz - 1] = 0;  // ensure null termination
    H5Tclose(mem_type);
    H5Tclose(atype);
    H5Aclose(attr);
}

// Read a 4-D double array [n_cell, NGLL, NGLL, NGLL]
static std::vector<double> read_4d_double(hid_t loc, const char* name, hsize_t& n_cell,
                                          hsize_t& ngll) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(ds);
    hsize_t dims[4];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n_cell = dims[0];
    ngll = dims[1];
    size_t total = (size_t)(dims[0] * dims[1] * dims[2] * dims[3]);
    std::vector<double> buf(total);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

// Read a 5-D double array [n_cell, NGLL, NGLL, NGLL, D]
static std::vector<double> read_5d_double(hid_t loc, const char* name, hsize_t& n_cell,
                                          hsize_t& ngll, hsize_t& last_dim) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    hid_t space = H5Dget_space(ds);
    hsize_t dims[5];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    n_cell = dims[0];
    ngll = dims[1];
    last_dim = dims[4];
    size_t total = (size_t)(dims[0] * dims[1] * dims[2] * dims[3] * dims[4]);
    std::vector<double> buf(total);
    H5Dread(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

// Read a 1-D int64 array
static std::vector<int64_t> read_int64_1d(hid_t loc, const char* name, hsize_t& n) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
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

// Write a 4-D double array
static void write_4d_double(hid_t loc, const char* name, hsize_t n_cell, hsize_t ngll,
                            const double* data) {
    hsize_t dims[4] = {n_cell, ngll, ngll, ngll};
    hid_t space = H5Screate_simple(4, dims, nullptr);
    hid_t ds =
        H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Sclose(space);
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: gf_preprocess_stage2 <model.h5>\n");
        return 1;
    }
    const char* model_path = argv[1];

    // Open file
    hid_t fid = open_or_fail(model_path, H5F_ACC_RDWR);

    // ---- Read topology ----
    hid_t topo_gid = H5Gopen2(fid, "topology", H5P_DEFAULT);
    int64_t n_cell;
    {
        hsize_t tmp;
        read_attr_int64(topo_gid, "n_cell", n_cell);
        if (n_cell <= 0) {
            // fallback: infer from dataset size
            std::vector<int64_t> dum = read_int64_1d(topo_gid, "cell_to_surface", tmp);
            n_cell = (int64_t)(tmp / 6);
        }
    }
    H5Gclose(topo_gid);

    // ---- Read config attrs ----
    hid_t cfg_gid = H5Gopen2(fid, "config", H5P_DEFAULT);
    double cfl_safety = 0.5, output_dt_s = 1.0, total_duration_s = 5.0;
    double storage_limit_gb = 50.0, record_depth_max_m = 5000.0;
    int64_t n_ranks = 1, nx_elements = 16, ny_elements = 16, ngll = 5;
    char snapshot_precision[16] = "float32";
    read_attr_double(cfg_gid, "cfl_safety", cfl_safety);
    read_attr_double(cfg_gid, "output_dt_s", output_dt_s);
    read_attr_double(cfg_gid, "total_duration_s", total_duration_s);
    read_attr_double(cfg_gid, "storage_limit_gb", storage_limit_gb);
    read_attr_double(cfg_gid, "record_depth_max_m", record_depth_max_m);
    read_attr_int64(cfg_gid, "n_ranks", n_ranks);
    read_attr_int64(cfg_gid, "nx_elements", nx_elements);
    read_attr_int64(cfg_gid, "ny_elements", ny_elements);
    read_attr_int64(cfg_gid, "NGLL", ngll);
    read_attr_str(cfg_gid, "snapshot_precision", snapshot_precision, sizeof(snapshot_precision));
    H5Gclose(cfg_gid);

    // ---- Read field/element arrays ----
    hid_t elem_gid = H5Gopen2(fid, "field/element", H5P_DEFAULT);

    hsize_t nc = 0, ng = 0, ld = 0;

    // Read coords [n_cell, NGLL, NGLL, NGLL, 3]
    std::vector<double> coords = read_5d_double(elem_gid, "coords", nc, ng, ld);
    if (nc != (hsize_t)n_cell || ng != (hsize_t)ngll) {
        fprintf(stderr, "ERROR: coords shape mismatch %llu %llu vs %lld %lld\n",
                (unsigned long long)nc, (unsigned long long)ng, (long long)n_cell,
                (long long)ngll);
        return 1;
    }

    // Read jacobian [n_cell, NGLL, NGLL, NGLL]
    hsize_t jnc = 0, jng = 0;
    std::vector<double> jacobian = read_4d_double(elem_gid, "jacobian", jnc, jng);

    // Read mass (geometry weights only), vp, vs, density
    std::vector<double> vp = read_4d_double(elem_gid, "vp", jnc, jng);
    std::vector<double> vs = read_4d_double(elem_gid, "vs", jnc, jng);
    std::vector<double> density = read_4d_double(elem_gid, "density", jnc, jng);
    std::vector<double> mass = read_4d_double(elem_gid, "mass", jnc, jng);

    H5Gclose(elem_gid);

    // ---- Read boundary_tag ----
    hid_t surf_gid = H5Gopen2(fid, "field/surface", H5P_DEFAULT);
    hsize_t ns = 0;
    std::vector<int64_t> boundary_tag = read_int64_1d(surf_gid, "boundary_tag", ns);
    H5Gclose(surf_gid);

    // ---- Compute λ, μ and density-weighted mass ----
    size_t n_total = (size_t)n_cell * (size_t)ngll * (size_t)ngll * (size_t)ngll;
    std::vector<double> lam(n_total), mu(n_total);
    for (size_t i = 0; i < n_total; ++i) {
        mu[i] = density[i] * vs[i] * vs[i];
        lam[i] = density[i] * (vp[i] * vp[i] - 2.0 * vs[i] * vs[i]);
        mass[i] *= density[i];
    }

    // ---- Compute solver_dt, snapshot_stride, nsteps ----
    // h_min from /info/ h_min attr (written by stage1)
    double h_min = 0, info_cfl_safety = 0;
    {
        hid_t info_gid = H5Gopen2(fid, "info", H5P_DEFAULT);
        read_attr_double(info_gid, "h_min", h_min);
        read_attr_double(info_gid, "cfl_safety", info_cfl_safety);
        H5Gclose(info_gid);
    }

    // Find vp_max
    double vp_max = 0;
    for (size_t i = 0; i < n_total; ++i)
        if (vp[i] > vp_max)
            vp_max = vp[i];

    double cfl_dt = (vp_max > 0 && h_min > 0) ? cfl_safety * h_min / vp_max : 0;
    double solver_dt = 0;
    int snapshot_stride = 1;
    if (cfl_dt > 0 && output_dt_s > 0) {
        for (int stride = 1; stride <= MAX_STRIDE; ++stride) {
            solver_dt = output_dt_s / stride;
            if (solver_dt <= cfl_dt) {
                snapshot_stride = stride;
                break;
            }
        }
    }
    if (solver_dt <= 0)
        solver_dt = cfl_dt;
    int64_t nsteps = (total_duration_s > 0 && solver_dt > 0)
                         ? (int64_t)std::ceil(total_duration_s / solver_dt)
                         : 0;

    // ---- Pre-flight statistics ----
    double detJ_min = 1e30, detJ_max = -1e30;
    double vp_min = 1e30, vs_min = 1e30, density_min = 1e30, lam_min = 1e30;
    double vp_max2 = -1e30, vs_max = -1e30, density_max = -1e30;

    for (size_t i = 0; i < n_total; ++i) {
        if (jacobian[i] < detJ_min)
            detJ_min = jacobian[i];
        if (jacobian[i] > detJ_max)
            detJ_max = jacobian[i];
        if (vp[i] < vp_min)
            vp_min = vp[i];
        if (vp[i] > vp_max2)
            vp_max2 = vp[i];
        if (vs[i] < vs_min)
            vs_min = vs[i];
        if (vs[i] > vs_max)
            vs_max = vs[i];
        if (density[i] < density_min)
            density_min = density[i];
        if (density[i] > density_max)
            density_max = density[i];
        if (lam[i] < lam_min)
            lam_min = lam[i];
    }

    int n_free = 0, n_absorbing = 0;
    for (hsize_t i = 0; i < ns; ++i) {
        if (boundary_tag[i] == 1)
            ++n_free;
        else if (boundary_tag[i] == 2)
            ++n_absorbing;
    }

    // Storage estimate
    int bytes_per = (std::strcmp(snapshot_precision, "float32") == 0) ? 4 : 8;
    int64_t n_snapshots =
        (snapshot_stride > 0) ? (nsteps + snapshot_stride - 1) / snapshot_stride : 0;
    int64_t n_gll_per_elem = ngll * ngll * ngll;
    double strain_one_run = (double)n_snapshots * n_cell * n_gll_per_elem * 6 * bytes_per;
    double restart_one_run = (double)n_cell * n_gll_per_elem * 3 * 3 * 8;
    double partition_est = (double)n_cell * n_gll_per_elem * 10 * 8;
    double total_gb = (strain_one_run * 3 + restart_one_run * 3 + partition_est) / 1e9;

    // ---- Write λ, μ to HDF5 ----
    hid_t fld_gid = H5Gopen2(fid, "field", H5P_DEFAULT);
    hid_t elem_wgid = H5Gopen2(fld_gid, "element", H5P_DEFAULT);

    if (H5Lexists(elem_wgid, "lambda", H5P_DEFAULT))
        H5Ldelete(elem_wgid, "lambda", H5P_DEFAULT);
    write_4d_double(elem_wgid, "lambda", (hsize_t)n_cell, (hsize_t)ngll, lam.data());

    if (H5Lexists(elem_wgid, "mu", H5P_DEFAULT))
        H5Ldelete(elem_wgid, "mu", H5P_DEFAULT);
    write_4d_double(elem_wgid, "mu", (hsize_t)n_cell, (hsize_t)ngll, mu.data());

    if (H5Lexists(elem_wgid, "mass", H5P_DEFAULT))
        H5Ldelete(elem_wgid, "mass", H5P_DEFAULT);
    write_4d_double(elem_wgid, "mass", (hsize_t)n_cell, (hsize_t)ngll, mass.data());

    H5Gclose(elem_wgid);
    H5Gclose(fld_gid);
    H5Fclose(fid);

    // ---- Print stats to stdout (machine-parseable for Python) ----
    printf("STAT_NCELL=%lld\n", (long long)n_cell);
    printf("STAT_NGLL=%lld\n", (long long)ngll);
    printf("STAT_NSTEPS=%lld\n", (long long)nsteps);
    printf("STAT_NSNAPSHOTS=%lld\n", (long long)n_snapshots);
    printf("STAT_SNAPSHOT_STRIDE=%d\n", snapshot_stride);
    printf("STAT_SOLVER_DT=%.15e\n", solver_dt);
    printf("STAT_CFL_DT=%.15e\n", cfl_dt);
    printf("STAT_DETJ_MIN=%.4e\n", detJ_min);
    printf("STAT_DETJ_MAX=%.4e\n", detJ_max);
    printf("STAT_VP_MIN=%.1f\n", vp_min);
    printf("STAT_VP_MAX=%.1f\n", vp_max2);
    printf("STAT_VS_MIN=%.1f\n", vs_min);
    printf("STAT_VS_MAX=%.1f\n", vs_max);
    printf("STAT_DENSITY_MIN=%.1f\n", density_min);
    printf("STAT_DENSITY_MAX=%.1f\n", density_max);
    printf("STAT_LAM_MIN=%.4e\n", lam_min);
    printf("STAT_N_FREE_SURFACES=%d\n", n_free);
    printf("STAT_N_ABSORBING_SURFACES=%d\n", n_absorbing);
    printf("STAT_ESTIMATED_STORAGE_GB=%.2f\n", total_gb);
    printf("STAT_SNAPSHOT_PRECISION=%s\n", snapshot_precision);
    printf("STAT_N_RANKS=%lld\n", (long long)n_ranks);
    fflush(stdout);

    fprintf(stderr, "Stage2 done: λ/μ, solver_dt=%.6e, nsteps=%lld, nsnapshots=%lld\n", solver_dt,
            (long long)nsteps, (long long)n_snapshots);
    return 0;
}