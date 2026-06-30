// tools/cpp/wavefield2vtk.cpp
// C++ accelerated wavefield2vtk — reads strain snapshots, writes per-step VTK.
// HDF5 is NOT thread-safe: snapshot loop is serial, per-vertex compute is parallel.

#include <dirent.h>
#include <omp.h>

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
                 "Usage: %s [--verbose] [--model MODEL.H5] [--config CONFIG.H5] [--vtk-dir DIR]\n",
                 prog);
}

struct RecordFile {
    std::string path;
    int rank;
};

static std::vector<RecordFile> find_record_files(const std::string& wave_dir) {
    std::vector<RecordFile> files;
    DIR* dir = opendir(wave_dir.c_str());
    if (!dir)
        return files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        std::regex re("record_(\\d+)\\.h5$");
        std::smatch m;
        if (std::regex_search(name, m, re) && m.size() > 1) {
            files.push_back({wave_dir + "/" + name, std::stoi(m[1].str())});
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end(),
              [](const RecordFile& a, const RecordFile& b) { return a.rank < b.rank; });
    return files;
}

int main(int argc, char** argv) {
    bool verbose = false;
    std::string model_path = "model.h5";
    std::string config_path = "config.h5";
    std::string vtk_dir = "vtk";

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
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    system(("mkdir -p " + vtk_dir).c_str());

    // ── Read mesh topology ──────────────────────────────────────
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
    if (dataset_exists(fm.id(), "field/element/is_pml")) {
        auto pml = read_int32_1d(fm.id(), "field/element/is_pml");
        for (size_t i = 0; i < pml.size() && i < (size_t)n_cell; ++i)
            is_pml[i] = (int8_t)pml[i];
    }
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

    // ── Find record files per direction ─────────────────────────
    std::map<std::string, std::vector<RecordFile>> record_paths;
    for (int di = 0; di < 3; ++di) {
        std::string wave_dir = "./wavefields/" + std::string(DIRECTIONS[di]);
        auto files = find_record_files(wave_dir);
        if (files.empty()) {
            std::cerr << "[wavefield2vtk] Error: no record_*.h5 files in " << wave_dir << "\n";
            return 1;
        }
        record_paths[DIRECTIONS[di]] = files;
        if (verbose)
            std::cout << "  wavefields/" << DIRECTIONS[di] << "/: " << files.size()
                      << " rank files\n";
    }

    // ── n_snapshots ─────────────────────────────────────────────
    int n_snapshots = 0;
    for (const auto& rf : record_paths["x"]) {
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
    std::cout << "  Snapshots: " << n_snapshots << "\n";

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

    // ── Pre-read vertex IDs + verify match across directions ────
    int n_ranks = (int)record_paths["x"].size();
    std::vector<std::vector<int64_t>> vertex_id_list(n_ranks);
    for (int ri = 0; ri < n_ranks; ++ri) {
        H5File f(record_paths["x"][ri].path);
        vertex_id_list[ri] = read_int64_1d(f.id(), "vertex_ids");
    }
    for (int di = 1; di < 3; ++di) {
        for (int ri = 0; ri < n_ranks; ++ri) {
            H5File f(record_paths[DIRECTIONS[di]][ri].path);
            auto vids = read_int64_1d(f.id(), "vertex_ids");
            if (vids != vertex_id_list[ri]) {
                std::cerr << "Error: vertex ID mismatch in "
                          << record_paths[DIRECTIONS[di]][ri].path << "\n";
                return 1;
            }
        }
    }

    // ── Open all record files, keep datasets open for snapshot loop ──
    // We keep file handles open (serial access, no thread-safety issue)
    struct OpenRec {
        hid_t fid;
        hid_t strain;
    };
    std::vector<std::vector<OpenRec>> open_files(3);
    for (int di = 0; di < 3; ++di) {
        for (const auto& rf : record_paths[DIRECTIONS[di]]) {
            hid_t fid = H5Fopen(rf.path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            if (fid < 0) {
                std::cerr << "Error opening " << rf.path << "\n";
                return 1;
            }
            hid_t sid = H5Dopen2(fid, "strain", H5P_DEFAULT);
            if (sid < 0) {
                std::cerr << "Error opening strain in " << rf.path << "\n";
                return 1;
            }
            open_files[di].push_back({fid, sid});
        }
    }

    // Pre-compute per-rank mapping
    struct RankMap {
        std::vector<int64_t> gids;
    };
    std::vector<RankMap> rank_maps(n_ranks);
    for (int ri = 0; ri < n_ranks; ++ri)
        rank_maps[ri].gids = vertex_id_list[ri];

    // Pre-compute vertex coords as float32
    std::vector<float> vtx_coords_f32(n_vert * 3);
    for (int64_t i = 0; i < n_vert; ++i) {
        vtx_coords_f32[i * 3 + 0] = (float)vertex_to_coord[i][0];
        vtx_coords_f32[i * 3 + 1] = (float)vertex_to_coord[i][1];
        vtx_coords_f32[i * 3 + 2] = (float)vertex_to_coord[i][2];
    }

    // Strain field names
    std::vector<std::string> strain_field_names;
    for (int di = 0; di < 3; ++di)
        for (int vi = 0; vi < 6; ++vi)
            strain_field_names.push_back(std::string("strain_") + VOIGT_LABELS[vi] + "_" +
                                         DIRECTIONS[di]);

    // ── Snapshot loop (serial — HDF5 not thread-safe) ──────────
    for (int snap_idx = 0; snap_idx < n_snapshots; ++snap_idx) {
        int step_num = snap_idx * stride;

        // Per-vertex accumulators
        std::vector<double> dir_strain[3];
        for (int di = 0; di < 3; ++di)
            dir_strain[di].assign(n_vert * 6, 0.0);
        std::vector<int> n_corners(n_vert, 0);

        // Read strain for this snapshot from each direction/rank (serial)
        for (int di = 0; di < 3; ++di) {
            for (int ri = 0; ri < n_ranks; ++ri) {
                int64_t nlv = (int64_t)rank_maps[ri].gids.size();
                if (nlv == 0)
                    continue;

                hid_t dset = open_files[di][ri].strain;
                hid_t fspace = H5Dget_space(dset);
                hsize_t start[3] = {(hsize_t)snap_idx, 0, 0};
                hsize_t count[3] = {1, (hsize_t)nlv, 6};
                H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count, nullptr);
                hid_t mspace = H5Screate_simple(3, count, nullptr);
                std::vector<double> sbuf(nlv * 6);
                H5Dread(dset, H5T_NATIVE_DOUBLE, mspace, fspace, H5P_DEFAULT, sbuf.data());
                H5Sclose(mspace);
                H5Sclose(fspace);

// Accumulate (this inner loop can be parallelized)
#pragma omp parallel for
                for (int64_t lvi = 0; lvi < nlv; ++lvi) {
                    int64_t gvid = rank_maps[ri].gids[lvi];
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

// Average per vertex (parallel)
#pragma omp parallel for
        for (int64_t vi = 0; vi < n_vert; ++vi) {
            if (n_corners[vi] > 0) {
                double inv = 1.0 / n_corners[vi];
                for (int di = 0; di < 3; ++di)
                    for (int c = 0; c < 6; ++c)
                        dir_strain[di][vi * 6 + c] *= inv;
            }
        }

        // Element corner average → cell strain (parallel)
        std::vector<double> elem_strain(3 * n_cell * 6, 0.0);
        std::vector<int> elem_nc(n_cell, 0);
#pragma omp parallel for
        for (int64_t ci = 0; ci < n_cell; ++ci) {
            int cnt = 0;
            for (int j = 0; j < 8; ++j) {
                int64_t gvid = connectivity[ci * 8 + j];
                if (gvid >= 0 && gvid < n_vert && n_corners[gvid] > 0) {
                    for (int di = 0; di < 3; ++di)
                        for (int c = 0; c < 6; ++c)
                            elem_strain[(di * n_cell + ci) * 6 + c] +=
                                dir_strain[di][gvid * 6 + c];
                    cnt++;
                }
            }
            elem_nc[ci] = cnt;
            if (cnt > 0) {
                double inv = 1.0 / cnt;
                for (int di = 0; di < 3; ++di)
                    for (int c = 0; c < 6; ++c)
                        elem_strain[(di * n_cell + ci) * 6 + c] *= inv;
            }
        }

        // Write VTK file for this snapshot
        char out_path[256];
        std::snprintf(out_path, sizeof(out_path), "%s/wavefield_%d.vtk", vtk_dir.c_str(),
                      step_num);
        if (verbose)
            std::cout << "[wavefield2vtk] Writing " << out_path << "\n";

        try {
            gf_vtk::VtkWriter vtk(out_path, "wavefield snapshot converted to VTK (C++)");
            vtk.write_points(vtx_coords_f32);

            int64_t n_hex = n_cell;
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
                    fd[ei] = (float)elem_strain[(di * n_cell + ei) * 6 + ci];
                vtk.write_scalar_field(strain_field_names[fi], fd);
            }

            std::vector<float> pml_f(n_hex);
            for (int64_t ei = 0; ei < n_hex; ++ei)
                pml_f[ei] = (float)is_pml[ei];
            vtk.write_scalar_field("PML_flag", pml_f);

            std::vector<float> nrec_f(n_hex);
            for (int64_t ei = 0; ei < n_hex; ++ei)
                nrec_f[ei] = (float)elem_nc[ei];
            vtk.write_scalar_field("n_recorded_corners", nrec_f);

        } catch (std::exception& e) {
            std::cerr << "  Error writing " << out_path << ": " << e.what() << "\n";
        }
    }

    // Cleanup
    for (int di = 0; di < 3; ++di)
        for (auto& of : open_files[di]) {
            H5Dclose(of.strain);
            H5Fclose(of.fid);
        }

    std::cout << "  Done. " << n_snapshots << " files written to " << vtk_dir << "/\n";
    return 0;
}