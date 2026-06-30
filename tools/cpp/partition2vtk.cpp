// tools/cpp/partition2vtk.cpp
// C++ accelerated partition2vtk — reads model.h5 + partitions/*.h5,
// writes per-rank VTK files with OpenMP parallelism.
// HDF5 is NOT thread-safe: all reads serial, VTK write parallel.

#include <dirent.h>
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "h5io.hh"
#include "topology.hh"
#include "vtk_writer.hh"

using namespace gf_topology;
using namespace gf_h5io;

static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: %s [--verbose] [--model MODEL.H5] [--vtk-dir DIR]\n", prog);
}

struct RankData {
    int rank;
    int64_t n_local;
    std::vector<int64_t> local_eids;
    std::vector<int64_t> local_conn;  // [n_local, 8]
    std::map<std::string, std::vector<float>> cell_fields;
};

int main(int argc, char** argv) {
    bool verbose = false;
    std::string model_path = "model.h5";
    std::string vtk_dir = "vtk";
    std::string part_dir = "partitions";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v")
            verbose = true;
        else if (arg == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (arg == "--vtk-dir" && i + 1 < argc)
            vtk_dir = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    system(("mkdir -p " + vtk_dir).c_str());

    // ── Read global topology ────────────────────────────────────
    std::cout << "[partition_to_vtk] Reading " << model_path << "\n";
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

    std::vector<int8_t> is_pml_global(n_cell, 0);
    if (dataset_exists(fm.id(), "field/element/is_pml")) {
        auto pml = read_int32_1d(fm.id(), "field/element/is_pml");
        for (size_t i = 0; i < pml.size() && i < (size_t)n_cell; ++i)
            is_pml_global[i] = (int8_t)pml[i];
    }

    std::cout << "  Global cells: " << n_cell << ", vertices: " << n_vert << "\n";

    // ── Scan partition files ────────────────────────────────────
    DIR* dir = opendir(part_dir.c_str());
    if (!dir) {
        std::cerr << "Error: cannot open " << part_dir << "\n";
        return 1;
    }
    struct dirent* entry;
    std::vector<std::string> part_files;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        if (name.size() > 3 && name.substr(0, 10) == "partition_" &&
            name.substr(name.size() - 3) == ".h5")
            part_files.push_back(name);
    }
    closedir(dir);
    std::sort(part_files.begin(), part_files.end());
    if (verbose)
        std::cout << "[partition_to_vtk] Found " << part_files.size() << " partition files\n";

    // ── Serial read: all partition data ─────────────────────────
    std::vector<float> vtx_coords(n_vert * 3);
    for (int64_t i = 0; i < n_vert; ++i) {
        vtx_coords[i * 3 + 0] = (float)vertex_to_coord[i][0];
        vtx_coords[i * 3 + 1] = (float)vertex_to_coord[i][1];
        vtx_coords[i * 3 + 2] = (float)vertex_to_coord[i][2];
    }

    std::vector<RankData> rank_data(part_files.size());

    for (size_t fi = 0; fi < part_files.size(); ++fi) {
        const auto& pf = part_files[fi];
        std::regex re("partition_(\\d+)\\.h5$");
        std::smatch m;
        int rank = -1;
        if (std::regex_search(pf, m, re) && m.size() > 1)
            rank = std::stoi(m[1].str());
        if (rank < 0)
            continue;

        auto& rd = rank_data[fi];
        rd.rank = rank;

        try {
            H5File fp(part_dir + "/" + pf);
            auto local_zero = read_int64_1d(fp.id(), "partition/local_element_ids");
            rd.local_eids.assign(local_zero.begin(), local_zero.end());
            rd.n_local = (int64_t)rd.local_eids.size();

            // Build local connectivity
            rd.local_conn.resize(rd.n_local * 8);
            for (int64_t li = 0; li < rd.n_local; ++li) {
                int64_t gid = rd.local_eids[li];
                auto conn = resolve_cell_vertices(cell_to_surface[gid].data(), surface_to_edge,
                                                  edge_to_vertex);
                for (int j = 0; j < 8; ++j)
                    rd.local_conn[li * 8 + j] = conn[j];
            }

            // Cell fields
            rd.cell_fields["PML_flag"].resize(rd.n_local);
            rd.cell_fields["Rank"].assign(rd.n_local, (float)rank);
            rd.cell_fields["Tile_Index"].assign(rd.n_local, -1.0f);
            for (int64_t li = 0; li < rd.n_local; ++li) {
                int64_t gid = rd.local_eids[li];
                rd.cell_fields["PML_flag"][li] = (float)is_pml_global[gid];
            }
            if (dataset_exists(fp.id(), "field/element/tile_index")) {
                auto tile = read_float64_1d(fp.id(), "field/element/tile_index");
                if ((int64_t)tile.size() == rd.n_local)
                    for (int64_t li = 0; li < rd.n_local; ++li)
                        rd.cell_fields["Tile_Index"][li] = (float)tile[li];
            }

            if (verbose)
                std::cout << "  [" << pf << "] " << rd.n_local << " elements\n";

        } catch (std::exception& e) {
            std::cerr << "Error reading " << pf << ": " << e.what() << "\n";
        }
    }

// ── Parallel VTK write ──────────────────────────────────────
#pragma omp parallel for
    for (size_t fi = 0; fi < rank_data.size(); ++fi) {
        const auto& rd = rank_data[fi];
        if (rd.rank < 0)
            continue;

        std::string out_path = vtk_dir + "/partition_" + std::to_string(rd.rank) + ".vtk";

        try {
            gf_vtk::VtkWriter vtk(out_path, "partition file converted to VTK (C++)");
            vtk.write_points(vtx_coords);

            // Cells: hex only (no GLL sub-cells in this simplified version)
            int64_t n_cells = rd.n_local;
            std::vector<int32_t> hex_cells(n_cells * 9);
            for (int64_t ci = 0; ci < n_cells; ++ci) {
                hex_cells[ci * 9] = 8;
                for (int j = 0; j < 8; ++j)
                    hex_cells[ci * 9 + 1 + j] = (int32_t)rd.local_conn[ci * 8 + j];
            }
            std::vector<int32_t> hex_types(n_cells, 12);
            vtk.write_cells(hex_cells, hex_types);

            // Cell data
            vtk.begin_cell_data(n_cells);
            for (const auto& kv : rd.cell_fields)
                vtk.write_scalar_field(kv.first, kv.second);

        } catch (std::exception& e) {
#pragma omp critical
            std::cerr << "Error writing " << out_path << ": " << e.what() << "\n";
        }
    }

    std::cout << "[partition_to_vtk] Done.\n";
    return 0;
}