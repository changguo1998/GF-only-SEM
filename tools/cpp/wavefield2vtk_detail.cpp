// tools/cpp/wavefield2vtk_detail.cpp
// C++ accelerated wavefield2vtk_detail — reads vertex strain snapshots,
// writes per-step VTK with strain as point data at mesh vertices.
//
// Unlike gf_wavefield2vtk (cell-corner average), this writes the raw
// per-vertex strain directly as 18 POINT_DATA fields (6 Voigt × 3 directions).

#include <dirent.h>
#include <omp.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
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
    int step;  // solver step number
};

// ── Record file discovery ──────────────────────────────────────────────

/// Find per-step record files matching record_{rank}_{step_num}.h5.
static std::vector<RecordFile> find_record_files(const std::string& wave_dir, int step_num) {
    std::vector<RecordFile> files;
    DIR* dir = opendir(wave_dir.c_str());
    if (!dir)
        return files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        std::regex re("record_(\\d+)_(\\d+)\\.h5$");
        std::smatch m;
        if (std::regex_search(name, m, re) && m.size() > 2) {
            int rank = std::stoi(m[1].str());
            int fstep = std::stoi(m[2].str());
            if (fstep == step_num) {
                files.push_back({wave_dir + "/" + name, rank, fstep});
            }
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end(),
              [](const RecordFile& a, const RecordFile& b) { return a.rank < b.rank; });
    return files;
}

/// Discover all unique solver step numbers from record files in a directory.
static std::vector<int> discover_steps(const std::string& wave_dir) {
    std::set<int> steps;
    DIR* dir = opendir(wave_dir.c_str());
    if (!dir)
        return {};
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        std::regex re("record_(\\d+)_(\\d+)\\.h5$");
        std::smatch m;
        if (std::regex_search(name, m, re) && m.size() > 2) {
            steps.insert(std::stoi(m[2].str()));
        }
    }
    closedir(dir);
    std::vector<int> sorted(steps.begin(), steps.end());
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

// ── HDF5 readers ──────────────────────────────────────────────────────

/// Read vertex_ids dataset from a record file.
static std::vector<int64_t> read_vertex_ids(const std::string& path) {
    H5File f(path);
    return read_int64_1d(f.id(), "vertex_ids");
}

/// Read strain[0, :, :] as float32 from a per-step record file.
/// Returns flattened [n_local * 6] float values.
static std::vector<float> read_step_strain(const std::string& path, int64_t n_local) {
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fid < 0)
        throw std::runtime_error("Cannot open: " + path);
    hid_t dset = H5Dopen2(fid, "strain", H5P_DEFAULT);
    if (dset < 0) {
        H5Fclose(fid);
        throw std::runtime_error("No strain dataset in " + path);
    }
    // Per-step files have shape [1, n_local, 6]
    std::vector<float> buf(n_local * 6);
    hsize_t count[3] = {1, (hsize_t)n_local, 6};
    hid_t mspace = H5Screate_simple(3, count, nullptr);
    H5Dread(dset, H5T_NATIVE_FLOAT, mspace, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Sclose(mspace);
    H5Dclose(dset);
    H5Fclose(fid);
    return buf;
}

/// Scatter per-rank strain into global vertex array.
/// strain_out: [n_vertex, 6] output (pre-allocated, caller zeros it).
static void scatter_strain(const std::vector<float>& rank_strain,
                           const std::vector<int64_t>& vertex_ids, int64_t n_vertex,
                           std::vector<float>& strain_out) {
    int64_t n_local = (int64_t)vertex_ids.size();
    for (int64_t li = 0; li < n_local; ++li) {
        int64_t gid = vertex_ids[li] - 1;  // 1-based → 0-based
        if (gid < 0 || gid >= n_vertex)
            continue;
        const float* src = rank_strain.data() + li * 6;
        float* dst = strain_out.data() + gid * 6;
        for (int c = 0; c < 6; ++c)
            dst[c] = src[c];
    }
}

// ── Main ──────────────────────────────────────────────────────────────

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

    // Create output directory
    struct stat st = {};
    if (stat(vtk_dir.c_str(), &st) != 0) {
        if (mkdir(vtk_dir.c_str(), 0755) != 0) {
            std::cerr << "Warning: could not create " << vtk_dir << "\n";
        }
    }

    // ── Read mesh topology ──────────────────────────────────────────
    if (verbose)
        std::cout << "[wavefield2vtk_detail] Reading " << model_path << "\n";
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

    // PML flag (optional)
    std::vector<int8_t> is_pml(n_cell, 0);
    if (dataset_exists(fm.id(), "field/element/is_pml")) {
        auto pml = read_int32_1d(fm.id(), "field/element/is_pml");
        for (size_t i = 0; i < pml.size() && i < (size_t)n_cell; ++i)
            is_pml[i] = (int8_t)pml[i];
    }
    if (verbose)
        std::cout << "  Global cells: " << n_cell << ", vertices: " << n_vert << "\n";

    // ── Resolve hex connectivity ─────────────────────────────────────
    if (verbose)
        std::cout << "[wavefield2vtk_detail] Resolving hex connectivity...\n";
    std::vector<int64_t> connectivity(n_cell * 8, -1);
    for (int64_t ci = 0; ci < n_cell; ++ci) {
        auto conn =
            resolve_cell_vertices(cell_to_surface[ci].data(), surface_to_edge, edge_to_vertex);
        for (int j = 0; j < 8; ++j)
            connectivity[ci * 8 + j] = conn[j];
    }

    // ── Read snapshot stride from config.h5 ──────────────────────────
    int stride = 1;
    try {
        H5File cf(config_path);
        stride = read_attr_int32_group(cf.id(), "simulation", "snapshot_stride");
    } catch (...) {
    }
    if (verbose)
        std::cout << "  Snapshot stride: " << stride << "\n";

    // ── Pre-compute vertex coords as float32 ─────────────────────────
    std::vector<float> vtx_coords_f32(n_vert * 3);
    for (int64_t i = 0; i < n_vert; ++i) {
        vtx_coords_f32[i * 3 + 0] = (float)vertex_to_coord[i][0];
        vtx_coords_f32[i * 3 + 1] = (float)vertex_to_coord[i][1];
        vtx_coords_f32[i * 3 + 2] = (float)vertex_to_coord[i][2];
    }

    // Strain field names: strain_{voigt}_{direction}
    std::vector<std::string> strain_field_names;
    for (int di = 0; di < 3; ++di)
        for (int vi = 0; vi < 6; ++vi)
            strain_field_names.push_back(std::string("strain_") + VOIGT_LABELS[vi] + "_" +
                                         DIRECTIONS[di]);

    // ── Discover available steps ─────────────────────────────────────
    std::vector<int> all_steps;
    if (snap_only >= 0) {
        all_steps.push_back(snap_only * stride);
    } else {
        all_steps = discover_steps("./wavefields/x/");
        if (all_steps.empty()) {
            std::cerr << "Error: no record files found in wavefields/x/\n";
            return 1;
        }
        if (verbose)
            std::cout << "  Found " << all_steps.size() << " snapshots\n";
    }
    int n_snapshots = (int)all_steps.size();

    // ── Pre-read vertex_ids from first step (validate across directions) ──
    std::vector<std::vector<int64_t>> vertex_id_list;
    {
        auto files = find_record_files("./wavefields/x/", all_steps[0]);
        if (files.empty()) {
            std::cerr << "Error: no record files for step " << all_steps[0] << "\n";
            return 1;
        }
        int n_ranks = (int)files.size();
        if (verbose)
            std::cout << "  Ranks: " << n_ranks << "\n";

        vertex_id_list.resize(n_ranks);
        for (int ri = 0; ri < n_ranks; ++ri) {
            vertex_id_list[ri] = read_vertex_ids(files[ri].path);
        }

        // Verify vertex_ids match across directions for rank 0
        for (int di = 1; di < 3; ++di) {
            std::string dwave = "./wavefields/" + std::string(DIRECTIONS[di]);
            auto dfiles = find_record_files(dwave, files[0].step);
            if (dfiles.empty()) {
                std::cerr << "Error: no files in " << dwave << "\n";
                return 1;
            }
            auto dvids = read_vertex_ids(dfiles[0].path);
            if (dvids != vertex_id_list[0]) {
                std::cerr << "Error: vertex_id mismatch in " << dwave << "\n";
                return 1;
            }
        }
        if (verbose)
            std::cout << "  Verified vertex IDs match across all directions\n";
    }

    // ── Process each snapshot ────────────────────────────────────────
    for (int si = 0; si < n_snapshots; ++si) {
        int step_num = all_steps[si];

        // Discover record files for this step across all directions
        std::vector<std::vector<RecordFile>> step_files(3);
        for (int di = 0; di < 3; ++di) {
            std::string wave_dir = "./wavefields/" + std::string(DIRECTIONS[di]);
            step_files[di] = find_record_files(wave_dir, step_num);
            if (step_files[di].empty()) {
                std::cerr << "Error: no record files for step " << step_num << " in " << wave_dir
                          << "\n";
                return 1;
            }
            std::sort(step_files[di].begin(), step_files[di].end(),
                      [](const RecordFile& a, const RecordFile& b) { return a.rank < b.rank; });
        }

        int n_ranks = (int)step_files[0].size();
        if (verbose)
            std::cout << "[wavefield2vtk_detail] Step " << step_num << " (" << n_ranks
                      << " rank(s))\n";

        // Per-direction vertex strain arrays [n_vert, 6]
        std::vector<float> dir_strain[3];
        for (int di = 0; di < 3; ++di)
            dir_strain[di].assign(n_vert * 6, 0.0f);

        // Read and scatter per direction, per rank
        for (int di = 0; di < 3; ++di) {
            for (int ri = 0; ri < n_ranks; ++ri) {
                int64_t n_local = (int64_t)vertex_id_list[ri].size();
                if (n_local == 0)
                    continue;
                auto sbuf = read_step_strain(step_files[di][ri].path, n_local);
                scatter_strain(sbuf, vertex_id_list[ri], n_vert, dir_strain[di]);
            }
        }

        // ── Write VTK ────────────────────────────────────────────────
        char out_path[256];
        std::snprintf(out_path, sizeof(out_path), "%s/wavefield_%d.vtk", vtk_dir.c_str(),
                      step_num);
        if (verbose)
            std::cout << "  Writing " << out_path << "\n";

        try {
            gf_vtk::VtkWriter vtk(out_path, "wavefield snapshot converted to VTK (C++ detail)");
            vtk.write_points(vtx_coords_f32);

            // Hex cells
            int64_t n_hex = n_cell;
            std::vector<int32_t> hex_cells(n_hex * 9);
            for (int64_t ci = 0; ci < n_hex; ++ci) {
                hex_cells[ci * 9] = 8;
                for (int j = 0; j < 8; ++j)
                    hex_cells[ci * 9 + 1 + j] = (int32_t)connectivity[ci * 8 + j];
            }
            std::vector<int32_t> hex_types(n_hex, 12);
            vtk.write_cells(hex_cells, hex_types);

            // Cell data: PML flag
            vtk.begin_cell_data(n_hex);
            std::vector<float> pml_f(n_hex);
            for (int64_t ei = 0; ei < n_hex; ++ei)
                pml_f[ei] = (float)is_pml[ei];
            vtk.write_scalar_field("PML_flag", pml_f);

            // Point data: 18 strain fields
            vtk.begin_point_data(n_vert);
            for (size_t fi = 0; fi < strain_field_names.size(); ++fi) {
                int di = (int)(fi / 6), ci = (int)(fi % 6);
                std::vector<float> pd(n_vert);
                for (int64_t vi = 0; vi < n_vert; ++vi)
                    pd[vi] = dir_strain[di][vi * 6 + ci];
                vtk.write_scalar_field(strain_field_names[fi], pd);
            }

        } catch (std::exception& e) {
            std::cerr << "  Error writing " << out_path << ": " << e.what() << "\n";
        }
    }

    if (verbose)
        std::cout << "  Done.\n";
    return 0;
}