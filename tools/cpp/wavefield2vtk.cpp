// tools/cpp/wavefield2vtk.cpp
// C++ accelerated wavefield2vtk — reads strain snapshots, writes per-step VTK.
//
// Default: process all snapshots (batch-read all).  With --snap N: process
// exactly snapshot N (read only that snapshot).  Combine with GNU parallel:
//     seq 0 499 | parallel -j16 OMP_NUM_THREADS=1 gf_wavefield2vtk --snap {}
//
// All HDF5 reads happen upfront (batch mode) or once (single-snap mode);
// the per-snapshot compute loops use OpenMP.

#include <dirent.h>
#include <omp.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "h5io.hh"
#include "topology.hh"
#include "vtk_writer.hh"

using namespace gf_topology;
using namespace gf_h5io;

static const char* VOIGT_LABELS[6] = {"xx", "yy", "zz", "xy", "xz", "yz"};
static const char* DIRECTIONS[3] = {"x", "y", "z"};

static void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --verbose, -v         Verbose output\n"
                 "  --model PATH          Model file (default: model.h5)\n"
                 "  --config PATH         Config file (default: config.h5)\n"
                 "  --vtk-dir DIR         Output directory (default: vtk)\n"
                 "  --snap N              Single snapshot index (default: all)\n"
                 "  --help, -h            This help\n",
                 prog);
}

struct RecordFile {
    std::string path;
    int rank;
    int step;  // solver step number (-1 if monolithic)
};

/// Find record files per direction.
/// If step_num >= 0, matches per-step files: record_{rank}_{step_num}.h5
/// If step_num < 0, matches monolithic files: record_{rank}.h5
static std::vector<RecordFile> find_record_files(const std::string& wave_dir, int step_num = -1) {
    std::vector<RecordFile> files;
    DIR* dir = opendir(wave_dir.c_str());
    if (!dir)
        return files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        if (step_num >= 0) {
            // Per-step files: record_{rank}_{step}.h5
            std::regex re("record_(\\d+)_(\\d+)\\.h5$");
            std::smatch m;
            if (std::regex_search(name, m, re) && m.size() > 2) {
                int rank = std::stoi(m[1].str());
                int fstep = std::stoi(m[2].str());
                if (fstep == step_num) {
                    files.push_back({wave_dir + "/" + name, rank, fstep});
                }
            }
        } else {
            // Monolithic files: record_{rank}.h5
            std::regex re("record_(\\d+)\\.h5$");
            std::smatch m;
            if (std::regex_search(name, m, re) && m.size() > 1) {
                files.push_back({wave_dir + "/" + name, std::stoi(m[1].str()), -1});
            }
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end(),
              [](const RecordFile& a, const RecordFile& b) { return a.rank < b.rank; });
    return files;
}

/// Read the full strain dataset from a per-step file (shape [1, nlv, 6] -> [nlv, 6]).
static std::vector<double> read_step_strain(const std::string& rpath, int64_t nlv) {
    hid_t fid = H5Fopen(rpath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0)
        throw std::runtime_error("Cannot open: " + rpath);
    hid_t dset = H5Dopen2(fid, "strain", H5P_DEFAULT);
    if (dset < 0) {
        H5Fclose(fid);
        throw std::runtime_error("No strain dataset in " + rpath);
    }
    std::vector<double> buf(nlv * 6);
    hsize_t count[3] = {1, (hsize_t)nlv, 6};
    hid_t mspace = H5Screate_simple(3, count, nullptr);
    H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Sclose(mspace);
    H5Dclose(dset);
    H5Fclose(fid);
    return buf;
}

/// Read all snapshots from an open dataset into a flat heap array.
static double* read_all_snapshots(hid_t dset, int64_t nlv, int n_snapshots) {
    double* buf = new double[n_snapshots * nlv * 6];
    hid_t fspace = H5Dget_space(dset);
    hsize_t start[3] = {0, 0, 0};
    hsize_t count[3] = {(hsize_t)n_snapshots, (hsize_t)nlv, 6};
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count, nullptr);
    hid_t mspace = H5Screate_simple(3, count, nullptr);
    H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, buf);
    H5Sclose(mspace);
    H5Sclose(fspace);
    return buf;
}

// ---------------------------------------------------------------------------
// GLL-format record file support
// ---------------------------------------------------------------------------

/// Read node IDs from a record file (gll_node_ids or legacy vertex_ids).
static std::vector<int64_t> read_node_ids(hid_t fid) {
    if (dataset_exists(fid, "gll_node_ids")) {
        return read_int64_1d(fid, "gll_node_ids");
    }
    return read_int64_1d(fid, "vertex_ids");
}

/// Read GLL node coordinates from a record file (if available).
/// Returns flattened [n_pts * 3] array.
static std::vector<double> read_node_coords(hid_t fid) {
    if (dataset_exists(fid, "gll_node_coords")) {
        std::vector<hsize_t> shape;
        auto flat = read_float64_nd(fid, "gll_node_coords", shape);
        return flat;  // already flattened by read_float64_nd
    }
    return {};
}

/// Check if record file uses GLL format (4D strain [snap, n_cell, n_node, 6]).
static bool is_gll_format(hid_t fid) {
    if (!dataset_exists(fid, "strain"))
        return false;
    hid_t dset = H5Dopen2(fid, "strain", H5P_DEFAULT);
    hid_t fspace = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(fspace);
    H5Sclose(fspace);
    H5Dclose(dset);
    return ndims == 4;
}

/// Read int32 dataset as flattened 1D vector (handles any dimensionality).
static std::vector<int32_t> read_int32_flat(hid_t fid, const std::string& name) {
    hid_t dset = H5Dopen2(fid, name.c_str(), H5P_DEFAULT);
    if (dset < 0)
        throw std::runtime_error("Cannot open: " + name);
    hid_t fspace = H5Dget_space(dset);
    int ndims = H5Sget_simple_extent_ndims(fspace);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(fspace, dims.data(), nullptr);
    int64_t total = 1;
    for (int i = 0; i < ndims; ++i)
        total *= (int64_t)dims[i];
    std::vector<int32_t> data(total);
    H5Dread(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(fspace);
    H5Dclose(dset);
    return data;
}

/// Read one snapshot of strain from GLL-format record file.
/// Returns per-GLL-node averaged strain [n_gll_nodes * 6].
static std::vector<double> read_step_strain_gll(const std::string& rpath, int64_t n_gll_nodes) {
    hid_t fid = H5Fopen(rpath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0)
        throw std::runtime_error("Cannot open: " + rpath);

    // Read cell_gll_node_index [n_cell, n_node]
    auto cgll = read_int32_flat(fid, "cell_gll_node_index");
    int64_t n_cell_node = (int64_t)cgll.size();
    int n_node_per_cell = 125;  // NGLL^3 = 5^3
    int64_t n_cell = n_cell_node / n_node_per_cell;

    // Read strain [1, n_cell, n_node, 6]
    hid_t dset = H5Dopen2(fid, "strain", H5P_DEFAULT);
    hid_t fspace = H5Dget_space(dset);
    hsize_t dims[4];
    H5Sget_simple_extent_dims(fspace, dims, nullptr);
    int64_t total_val = (int64_t)dims[1] * dims[2] * dims[3];
    std::vector<double> raw(total_val);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, fspace, H5P_DEFAULT, raw.data());
    H5Sclose(fspace);
    H5Dclose(dset);
    H5Fclose(fid);

    // Average per GLL node
    std::vector<double> node_strain(n_gll_nodes * 6, 0.0);
    std::vector<int> count(n_gll_nodes, 0);

    for (int64_t c = 0; c < n_cell; ++c) {
        for (int nn = 0; nn < n_node_per_cell; ++nn) {
            int32_t gll_idx = cgll[c * n_node_per_cell + nn];
            if (gll_idx < 0 || gll_idx >= n_gll_nodes)
                continue;
            int64_t raw_off = (c * n_node_per_cell + nn) * 6;
            for (int vi = 0; vi < 6; ++vi) {
                double val = raw[raw_off + vi];
                if (std::isfinite(val)) {
                    node_strain[gll_idx * 6 + vi] += val;
                    if (vi == 0)
                        count[gll_idx]++;
                }
            }
        }
    }

    for (int64_t i = 0; i < n_gll_nodes; ++i) {
        if (count[i] > 0) {
            double inv = 1.0 / count[i];
            for (int vi = 0; vi < 6; ++vi)
                node_strain[i * 6 + vi] *= inv;
        }
    }

    return node_strain;
}

/// Read all snapshots from GLL-format record file.
/// Returns heap array [n_snapshots * n_gll_nodes * 6].
static double* read_all_snapshots_gll(hid_t fid, int64_t n_gll_nodes, int n_snapshots) {
    // Read cell_gll_node_index
    auto cgll = read_int32_flat(fid, "cell_gll_node_index");
    int n_node_per_cell = 125;
    int64_t n_cell = (int64_t)cgll.size() / n_node_per_cell;

    // Read full strain [n_snapshots, n_cell, n_node, 6]
    hid_t dset = H5Dopen2(fid, "strain", H5P_DEFAULT);
    hid_t fspace = H5Dget_space(dset);
    hsize_t dims[4];
    H5Sget_simple_extent_dims(fspace, dims, nullptr);
    int64_t total = (int64_t)dims[0] * dims[1] * dims[2] * dims[3];
    std::vector<double> raw(total);
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, fspace, H5P_DEFAULT, raw.data());
    H5Sclose(fspace);
    H5Dclose(dset);

    double* buf = new double[n_snapshots * n_gll_nodes * 6];
    std::fill(buf, buf + n_snapshots * n_gll_nodes * 6, 0.0);
    std::vector<int> count(n_gll_nodes, 0);

    // Process each snapshot
    for (int s = 0; s < n_snapshots; ++s) {
        std::fill(count.begin(), count.end(), 0);
        double* snap_buf = buf + s * n_gll_nodes * 6;

        for (int64_t c = 0; c < n_cell; ++c) {
            for (int nn = 0; nn < n_node_per_cell; ++nn) {
                int32_t gll_idx = cgll[c * n_node_per_cell + nn];
                if (gll_idx < 0 || gll_idx >= n_gll_nodes)
                    continue;
                int64_t raw_off = ((int64_t)s * n_cell + c) * n_node_per_cell * 6 + nn * 6;
                for (int vi = 0; vi < 6; ++vi) {
                    double val = raw[raw_off + vi];
                    if (std::isfinite(val)) {
                        snap_buf[gll_idx * 6 + vi] += val;
                        if (vi == 0)
                            count[gll_idx]++;
                    }
                }
            }
        }

        for (int64_t i = 0; i < n_gll_nodes; ++i) {
            if (count[i] > 0) {
                double inv = 1.0 / count[i];
                for (int vi = 0; vi < 6; ++vi)
                    snap_buf[i * 6 + vi] *= inv;
            }
        }
    }

    return buf;
}

/// Process a single snapshot: read data, accumulate, average, write VTK.
static void process_snapshot(
    int snap_idx, int step_num, const std::vector<std::vector<RecordFile>>& record_paths,
    const std::vector<std::vector<int64_t>>& vertex_id_list, int n_vert, int64_t n_cell_local,
    const std::vector<int64_t>& connectivity, const std::vector<int8_t>& is_pml,
    const std::vector<float>& vtx_coords_f32, const std::vector<std::string>& strain_field_names,
    const std::string& vtk_dir, const std::string& rank_suffix, bool verbose,
    // Output: per-rank dir_strain accumulators filled from rank files,
    // then averaged per-vertex and per-cell.
    // For --snap mode, data is read here.
    // For batch mode, data comes from pre-read buffers.
    // We differentiate by checking if batch_di_ri is null.
    bool batch_mode,
    const std::vector<std::vector<const double*>>& batch_data,  // [di][ri], nullptr if not batch
    const std::vector<std::vector<int64_t>>& batch_nlv) {
    // Per-vertex accumulators
    std::vector<double> dir_strain[3];
    for (int di = 0; di < 3; ++di)
        dir_strain[di].assign(n_vert * 6, 0.0);
    std::vector<int> n_corners(n_vert, 0);

    int n_ranks = (int)record_paths[0].size();

    for (int di = 0; di < 3; ++di) {
        for (int ri = 0; ri < n_ranks; ++ri) {
            int64_t nlv = (int64_t)vertex_id_list[ri].size();
            if (nlv == 0)
                continue;

            // Get data pointer (batch or freshly read)
            std::vector<double> local_buf;
            const double* sbuf = nullptr;
            if (batch_mode && batch_data[di][ri] != nullptr) {
                sbuf = batch_data[di][ri] + snap_idx * nlv * 6;
            } else {
                std::string rpath = record_paths[di][ri].path;
                hid_t check_fid = H5Fopen(rpath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                bool gll_fmt = is_gll_format(check_fid);
                H5Fclose(check_fid);
                if (gll_fmt) {
                    local_buf = read_step_strain_gll(rpath, nlv);
                } else {
                    local_buf = read_step_strain(rpath, nlv);
                }
                sbuf = local_buf.data();
            }

#pragma omp parallel for
            for (int64_t lvi = 0; lvi < nlv; ++lvi) {
                int64_t gvid = vertex_id_list[ri][lvi];
                if (gvid < 0 || gvid >= n_vert)
                    continue;
                for (int c = 0; c < 6; ++c) {
                    double val = sbuf[lvi * 6 + c];
                    if (std::isfinite(val)) {
#pragma omp atomic
                        dir_strain[di][gvid * 6 + c] += val;
                        if (di == 0) {
#pragma omp atomic
                            n_corners[gvid]++;
                        }
                    }
                }
            }
        }
    }

    // Average per vertex
#pragma omp parallel for
    for (int64_t vi = 0; vi < n_vert; ++vi) {
        if (n_corners[vi] > 0) {
            double inv = 1.0 / n_corners[vi];
            for (int di = 0; di < 3; ++di)
                for (int c = 0; c < 6; ++c)
                    dir_strain[di][vi * 6 + c] *= inv;
        }
    }

    // Element corner average → cell strain
    std::vector<double> cell_strain(3 * n_cell_local * 6, 0.0);
    std::vector<int> cell_nc(n_cell_local, 0);
#pragma omp parallel for
    for (int64_t ci = 0; ci < n_cell_local; ++ci) {
        int cnt = 0;
        for (int j = 0; j < 8; ++j) {
            int64_t gvid = connectivity[ci * 8 + j];
            if (gvid >= 0 && gvid < n_vert && n_corners[gvid] > 0) {
                for (int di = 0; di < 3; ++di)
                    for (int c = 0; c < 6; ++c)
                        cell_strain[(di * n_cell_local + ci) * 6 + c] +=
                            dir_strain[di][gvid * 6 + c];
                cnt++;
            }
        }
        cell_nc[ci] = cnt;
        if (cnt > 0) {
            double inv = 1.0 / cnt;
            for (int di = 0; di < 3; ++di)
                for (int c = 0; c < 6; ++c)
                    cell_strain[(di * n_cell_local + ci) * 6 + c] *= inv;
        }
    }

    // Write VTK file
    char out_path[256];
    std::snprintf(out_path, sizeof(out_path), "%s/wavefield_%d%s.vtk", vtk_dir.c_str(), step_num,
                  rank_suffix.c_str());
    if (verbose)
        std::cout << "[wavefield2vtk] Writing " << out_path << "\n";

    try {
        gf_vtk::VtkWriter vtk(out_path, "wavefield snapshot converted to VTK (C++)");
        vtk.write_points(vtx_coords_f32);

        int64_t n_hex = n_cell_local;
        std::vector<int32_t> hex_cells(n_hex * 9);
#pragma omp parallel for
        for (int64_t ci = 0; ci < n_hex; ++ci) {
            hex_cells[ci * 9] = 8;
            for (int j = 0; j < 8; ++j)
                hex_cells[ci * 9 + 1 + j] = (int32_t)connectivity[ci * 8 + j];
        }
        std::vector<int32_t> hex_types(n_hex, 12);
        vtk.write_cells(hex_cells, hex_types);

        vtk.begin_cell_data(n_hex);
        for (size_t fi = 0; fi < strain_field_names.size(); ++fi) {
            int di = (int)(fi / 6), ci = (int)(fi % 6);
            std::vector<float> fd(n_hex);
            for (int64_t ei = 0; ei < n_hex; ++ei)
                fd[ei] = (float)cell_strain[(di * n_cell_local + ei) * 6 + ci];
            vtk.write_scalar_field(strain_field_names[fi], fd);
        }

        std::vector<float> pml_f(n_hex);
        for (int64_t ei = 0; ei < n_hex; ++ei)
            pml_f[ei] = (float)is_pml[ei];
        vtk.write_scalar_field("PML_flag", pml_f);

        std::vector<float> nrec_f(n_hex);
        for (int64_t ei = 0; ei < n_hex; ++ei)
            nrec_f[ei] = (float)cell_nc[ei];
        vtk.write_scalar_field("n_recorded_corners", nrec_f);

    } catch (std::exception& e) {
        std::cerr << "  Error writing " << out_path << ": " << e.what() << "\n";
    }
}

int main(int argc, char** argv) {
    bool verbose = false;
    std::string model_path = "model.h5";
    std::string config_path = "config.h5";
    std::string vtk_dir = "vtk";
    int snap_only = -1;  // -1 = process all snapshots

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v")
            verbose = true;
        else if (arg == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (arg == "--config" && i + 1 < argc)
            config_path = argv[++i];
        else if (arg == "--vtk-dir" && i + 1 < argc)
            vtk_dir = argv[++i];
        else if (arg == "--snap" && i + 1 < argc)
            snap_only = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    struct stat st = {};
    if (stat(vtk_dir.c_str(), &st) != 0) {
        if (mkdir(vtk_dir.c_str(), 0755) != 0) {
            std::cerr << "Warning: could not create " << vtk_dir << "\n";
        }
    }

    // ── Read mesh topology ──────────────────────────────────────
    if (verbose)
        std::cout << "[wavefield2vtk] Reading " << model_path << "\n";
    H5File fm(model_path);
    hid_t topo_gid = H5Gopen2(fm.id(), "topology", H5P_DEFAULT);
    if (topo_gid < 0) {
        std::cerr << "Error: no /topology group\n";
        return 1;
    }
    auto vertex_to_coord = read_float64_2d(topo_gid, "vertex_to_coord");
    int64_t n_vert = (int64_t)vertex_to_coord.size();
    auto edge_to_vertex = read_int64_2d(topo_gid, "edge_to_vertex");
    auto surface_to_edge = read_int64_2d(topo_gid, "surface_to_edge");
    auto cell_to_surface = read_int64_2d(topo_gid, "cell_to_surface");
    H5Gclose(topo_gid);
    int64_t n_cell = (int64_t)cell_to_surface.size();

    std::vector<int8_t> is_pml(n_cell, 0);
    if (dataset_exists(fm.id(), "field/cell/is_pml")) {
        auto pml = read_int32_1d(fm.id(), "field/cell/is_pml");
        for (size_t i = 0; i < pml.size() && i < (size_t)n_cell; ++i)
            is_pml[i] = (int8_t)pml[i];
    }
    if (verbose)
        std::cout << "  Global cells: " << n_cell << ", vertices: " << n_vert << "\n";

    // ── Resolve connectivity ────────────────────────────────────
    if (verbose)
        std::cout << "[wavefield2vtk] Resolving hexahedral connectivity...\n";
    std::vector<int64_t> connectivity(n_cell * 8, -1);
    for (int64_t ci = 0; ci < n_cell; ++ci) {
        auto conn =
            resolve_cell_vertices(cell_to_surface[ci].data(), surface_to_edge, edge_to_vertex);
        for (int j = 0; j < 8; ++j)
            connectivity[ci * 8 + j] = conn[j];
    }

    // ── Read snapshot stride from config.h5 (needed before file discovery) ──
    int stride = 1;
    try {
        hid_t cf = H5Fopen(config_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (cf >= 0) {
            stride = read_attr_int32_group(cf, "simulation", "snapshot_stride");
            H5Fclose(cf);
        }
    } catch (...) {
    }
    if (verbose)
        std::cout << "  Snapshot stride: " << stride << "\n";

    // ── Find record files per direction ─────────────────────────
    std::vector<std::vector<RecordFile>> record_paths(3);
    int n_snapshots = 0;
    if (snap_only >= 0) {
        // Per-step file mode (--snap N): find record_{rank}_{step}.h5
        int step_num = snap_only * stride;
        for (int di = 0; di < 3; ++di) {
            std::string wave_dir = "./wavefields/" + std::string(DIRECTIONS[di]);
            auto files = find_record_files(wave_dir, step_num);
            if (files.empty()) {
                std::cerr << "[wavefield2vtk] Error: no record_*_" << step_num << ".h5 files in "
                          << wave_dir << "\n";
                return 1;
            }
            record_paths[di] = files;
            if (verbose)
                std::cout << "  wavefields/" << DIRECTIONS[di] << "/: " << files.size()
                          << " rank files (step=" << step_num << ")\n";
        }
        n_snapshots = 1;  // each per-step file has 1 snapshot
    } else {
        // Monolithic file mode (batch): find record_{rank}.h5
        for (int di = 0; di < 3; ++di) {
            std::string wave_dir = "./wavefields/" + std::string(DIRECTIONS[di]);
            auto files = find_record_files(wave_dir);  // step_num = -1
            if (files.empty()) {
                std::cerr << "[wavefield2vtk] Error: no record_*.h5 files in " << wave_dir << "\n";
                return 1;
            }
            record_paths[di] = files;
            if (verbose)
                std::cout << "  wavefields/" << DIRECTIONS[di] << "/: " << files.size()
                          << " rank files\n";
        }

        // Determine n_snapshots from first file
        for (const auto& rf : record_paths[0]) {
            H5File f(rf.path);
            if (dataset_exists(f.id(), "strain")) {
                std::vector<hsize_t> sh;
                read_float64_nd(f.id(), "strain", sh);
                if (!sh.empty() && sh[0] > 0) {
                    n_snapshots = (int)sh[0];
                    break;
                }
            }
        }
        if (n_snapshots == 0) {
            std::cerr << "Error: no snapshots\n";
            return 1;
        }

        if (snap_only >= 0 && snap_only >= n_snapshots) {
            std::cerr << "Error: --snap " << snap_only << " out of range (0.." << n_snapshots - 1
                      << ")\n";
            return 1;
        }
    }
    if (verbose)
        std::cout << "  Snapshots: " << n_snapshots << (snap_only >= 0 ? " (--snap mode)" : "")
                  << "\n";

    // ── Pre-read vertex IDs + verify match across directions ────
    int n_ranks = (int)record_paths[0].size();
    std::vector<std::vector<int64_t>> vertex_id_list(n_ranks);
    for (int ri = 0; ri < n_ranks; ++ri) {
        H5File f(record_paths[0][ri].path);
        vertex_id_list[ri] = read_node_ids(f.id());
    }
    for (int di = 1; di < 3; ++di) {
        for (int ri = 0; ri < n_ranks; ++ri) {
            H5File f(record_paths[di][ri].path);
            auto vids = read_node_ids(f.id());
            if (vids != vertex_id_list[ri]) {
                std::cerr << "Error: vertex ID mismatch in " << record_paths[di][ri].path << "\n";
                return 1;
            }
        }
    }

    // ── Pre-compute point coords as float32 ─────────────────────
    // For GLL-format records, use gll_node_coords from record files.
    // For legacy records, use vertex_to_coord from model.h5.
    bool use_gll_coords = false;
    std::vector<float> vtx_coords_f32;
    {
        H5File f0(record_paths[0][0].path);
        auto gll_coords = read_node_coords(f0.id());
        if (!gll_coords.empty()) {
            use_gll_coords = true;
            int64_t n_pts = (int64_t)gll_coords.size() / 3;
            vtx_coords_f32.resize(n_pts * 3);
            for (int64_t i = 0; i < n_pts; ++i) {
                vtx_coords_f32[i * 3 + 0] = (float)gll_coords[i * 3 + 0];
                vtx_coords_f32[i * 3 + 1] = (float)gll_coords[i * 3 + 1];
                vtx_coords_f32[i * 3 + 2] = (float)gll_coords[i * 3 + 2];
            }
            n_vert = n_pts;  // override n_vert with GLL node count
            if (verbose)
                std::cout << "  Using GLL node coords: " << n_pts << " points\n";
        }
    }
    if (!use_gll_coords) {
        vtx_coords_f32.resize(n_vert * 3);
        for (int64_t i = 0; i < n_vert; ++i) {
            vtx_coords_f32[i * 3 + 0] = (float)vertex_to_coord[i][0];
            vtx_coords_f32[i * 3 + 1] = (float)vertex_to_coord[i][1];
            vtx_coords_f32[i * 3 + 2] = (float)vertex_to_coord[i][2];
        }
    }

    // Strain field names
    std::vector<std::string> strain_field_names;
    for (int di = 0; di < 3; ++di)
        for (int vi = 0; vi < 6; ++vi)
            strain_field_names.push_back(std::string("strain_") + VOIGT_LABELS[vi] + "_" +
                                         DIRECTIONS[di]);

    // ── Determine snapshot range ────────────────────────────────
    int snap_start = (snap_only >= 0) ? snap_only : 0;
    int snap_end = (snap_only >= 0) ? snap_only + 1 : n_snapshots;

    // ── Decide mode: batch-read all vs single-snapshot ──────────
    bool batch_mode = (snap_only < 0);  // batch-read all snapshots only in full mode

    // For batch mode: pre-read all snapshots
    // [di][ri] → heap buffer [n_snapshots * nlv * 6]
    std::vector<std::vector<double*>> batch_data(3, std::vector<double*>(n_ranks, nullptr));
    std::vector<std::vector<int64_t>> batch_nlv(3, std::vector<int64_t>(n_ranks, 0));

    if (batch_mode) {
        if (verbose)
            std::cout << "[wavefield2vtk] Batch-reading all snapshots...\n";
        for (int di = 0; di < 3; ++di) {
            for (int ri = 0; ri < n_ranks; ++ri) {
                int64_t nlv = (int64_t)vertex_id_list[ri].size();
                batch_nlv[di][ri] = nlv;
                if (nlv == 0)
                    continue;
                std::string rpath = record_paths[di][ri].path;
                hid_t fid = H5Fopen(rpath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
                if (is_gll_format(fid)) {
                    batch_data[di][ri] = read_all_snapshots_gll(fid, nlv, n_snapshots);
                    H5Fclose(fid);
                } else {
                    hid_t dset = H5Dopen2(fid, "strain", H5P_DEFAULT);
                    batch_data[di][ri] = read_all_snapshots(dset, nlv, n_snapshots);
                    H5Dclose(dset);
                    H5Fclose(fid);
                }
            }
        }
    }

    // ── Build view arrays for process_snapshot ──────────────────
    // batch mode: const double* ptrs into pre-read buffers
    // single-snap mode: null ptrs (process_snapshot reads on-demand)
    std::vector<std::vector<const double*>> batch_views(
        3, std::vector<const double*>(n_ranks, nullptr));
    if (batch_mode) {
        for (int di = 0; di < 3; ++di)
            for (int ri = 0; ri < n_ranks; ++ri)
                batch_views[di][ri] = batch_data[di][ri];
    }

    // ── Process snapshots ───────────────────────────────────────
    for (int si = snap_start; si < snap_end; ++si) {
        int step_num = si * stride;
        process_snapshot(si, step_num, record_paths, vertex_id_list, (int)n_vert, n_cell,
                         connectivity, is_pml, vtx_coords_f32, strain_field_names, vtk_dir, "",
                         verbose, batch_mode, batch_views, batch_nlv);
    }

    // ── Cleanup batch data ──
    if (batch_mode) {
        for (int di = 0; di < 3; ++di)
            for (int ri = 0; ri < n_ranks; ++ri)
                delete[] batch_data[di][ri];
    }

    if (verbose)
        std::cout << "  Done. " << (snap_end - snap_start) << " files written to " << vtk_dir
                  << "/\n";
    return 0;
}