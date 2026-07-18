// tools/cpp/model2vtk.cpp
// C++ accelerated model2vtk — reads model.h5, writes model.vtk
// Uses OpenMP for parallel field assembly and cell connectivity resolution.
// Keep Python edition in tools/model2vtk.py for reference.

#include <omp.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "h5io.hh"
#include "topology.hh"
#include "vtk_writer.hh"

using namespace gf_topology;
using namespace gf_h5io;

// ── CLI ────────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    std::fprintf(stderr, "Usage: %s [--verbose] [--model MODEL.H5] [--vtk-dir DIR]\n", prog);
    std::fprintf(stderr, "  Reads model.h5 (and partitions/ if present), writes vtk/model.vtk\n");
}

// ── Mesh vertex interpolation (cell-averaged → vertex) ────────────
struct VertexToCell {
    std::vector<int32_t> v2c;
    std::vector<int32_t> offsets;  // size n_vert+1
};

static VertexToCell build_vertex_to_cell(const std::vector<int64_t>& connectivity_flat,
                                         int64_t n_cell, int64_t n_vert) {
    VertexToCell vtc;
    std::vector<int32_t> counts(n_vert, 0);
    for (int64_t ci = 0; ci < n_cell; ++ci) {
        for (int j = 0; j < 8; ++j) {
            int64_t v = connectivity_flat[ci * 8 + j];
            if (v >= 0 && v < n_vert)
                counts[v]++;
        }
    }
    vtc.offsets.resize(n_vert + 1, 0);
    int32_t sum = 0;
    for (int64_t vi = 0; vi < n_vert; ++vi) {
        vtc.offsets[vi] = sum;
        sum += counts[vi];
    }
    vtc.offsets[n_vert] = sum;
    vtc.v2c.resize(sum, -1);
    std::vector<int32_t> cur(n_vert, 0);
    for (int64_t ci = 0; ci < n_cell; ++ci) {
        for (int j = 0; j < 8; ++j) {
            int64_t v = connectivity_flat[ci * 8 + j];
            if (v >= 0 && v < n_vert) {
                int32_t pos = vtc.offsets[v] + cur[v];
                vtc.v2c[pos] = (int32_t)ci;
                cur[v]++;
            }
        }
    }
    return vtc;
}

// Interpolate cell-centered field onto mesh vertices.
static std::vector<float> interpolate_mesh_vertex_field(const std::vector<double>& cell_field,
                                                        const VertexToCell& vtc, int64_t n_vert) {
    std::vector<float> result(n_vert, 0.0f);
#pragma omp parallel for
    for (int64_t vi = 0; vi < n_vert; ++vi) {
        int32_t start = vtc.offsets[vi];
        int32_t end = vtc.offsets[vi + 1];
        if (end > start) {
            double sum = 0.0;
            for (int32_t p = start; p < end; ++p) {
                sum += cell_field[vtc.v2c[p]];
            }
            result[vi] = (float)(sum / (double)(end - start));
        }
    }
    return result;
}

// ── Main ───────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    bool verbose = false;
    std::string model_path = "model.h5";
    std::string vtk_dir = "vtk";
    std::string config_path = "config.h5";
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
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + vtk_dir;
    // mkdir -p already created vtk_dir
    // If mkdir fails, we will catch it when trying to write the file
    struct stat st = {};
    if (stat(vtk_dir.c_str(), &st) != 0) {
        if (mkdir(vtk_dir.c_str(), 0755) != 0) {
            std::cerr << "Warning: could not create " << vtk_dir << "\n";
        }
    }

    bool has_partitions = false;
    {
        H5File f(model_path);
        has_partitions = dataset_exists(f.id(), "partitions");
    }

    // ── Read topology ────────────────────────────────────────────
    std::cout << "[model_to_vtk] Reading " << model_path << "\n";

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

    // Read is_pml
    std::vector<int8_t> is_pml(n_cell, 0);
    if (dataset_exists(fm.id(), "field/cell/is_pml")) {
        auto pml_data = read_int32_1d(fm.id(), "field/cell/is_pml");
        for (size_t i = 0; i < pml_data.size() && i < (size_t)n_cell; ++i)
            is_pml[i] = (int8_t)pml_data[i];
    }

    // Read GLL coords
    bool has_gll = dataset_exists(fm.id(), "field/cell/coords");
    std::vector<hsize_t> gll_shape;
    std::vector<double> gll_coords_flat;
    int ngll = 0;
    if (has_gll) {
        gll_coords_flat = read_float64_nd(fm.id(), "field/cell/coords", gll_shape);
        if (gll_shape.size() >= 4) {
            ngll = (int)gll_shape[1];  // [n_cell, ngll, ngll, ngll, 3]
        }
        if (verbose)
            std::cout << "  GLL coords: ngll=" << ngll << "\n";
    }

    std::cout << "  Cells: " << n_cell << ", Vertices: " << n_vert << "\n";

    // ── Resolve connectivity ─────────────────────────────────────
    if (verbose)
        std::cout << "[model_to_vtk] Resolving hexahedral connectivity...\n";
    std::vector<int64_t> connectivity(n_cell * 8, -1);
#pragma omp parallel for
    for (int64_t ci = 0; ci < n_cell; ++ci) {
        auto conn =
            resolve_cell_vertices(cell_to_surface[ci].data(), surface_to_edge, edge_to_vertex);
        for (int j = 0; j < 8; ++j)
            connectivity[ci * 8 + j] = conn[j];
    }

    // Convert vertex coords to float32 for VTK
    std::vector<float> vtx_coords(n_vert * 3);
    for (int64_t i = 0; i < n_vert; ++i) {
        vtx_coords[i * 3 + 0] = (float)vertex_to_coord[i][0];
        vtx_coords[i * 3 + 1] = (float)vertex_to_coord[i][1];
        vtx_coords[i * 3 + 2] = (float)vertex_to_coord[i][2];
    }

    // ── Cell fields ──────────────────────────────────────────────
    std::map<std::string, std::vector<float>> cell_fields;
    cell_fields["PML_flag"].resize(n_cell);
    for (int64_t i = 0; i < n_cell; ++i)
        cell_fields["PML_flag"][i] = (float)is_pml[i];

    cell_fields["Tile_Index"].assign(n_cell, -1.0f);
    // Try to read tile_index from model.h5
    if (dataset_exists(fm.id(), "field/cell/tile_index")) {
        auto tile_raw = read_int64_1d(fm.id(), "field/cell/tile_index");
        for (size_t i = 0; i < tile_raw.size() && i < (size_t)n_cell; ++i)
            cell_fields["Tile_Index"][i] = (float)tile_raw[i];
    }

    // ── Point fields & GLL detail ────────────────────────────────
    std::map<std::string, std::vector<float>> point_fields;
    std::vector<float> vertex_coords_out = vtx_coords;
    int64_t n_mesh_vert = n_vert;

    GllTemplate gll_tmpl;
    GllCells gll_cells;
    std::vector<int32_t> full_cell_types;
    std::vector<int32_t> full_cells;
    int64_t n_gll_cells = 0;

    if (has_gll && ngll > 0) {
        int64_t gll_per_cell = (int64_t)ngll * ngll * ngll;
        // Read GLL point fields
        std::vector<std::string> gll_field_names = {"vp", "vs", "density", "mass", "damping"};
        std::map<std::string, std::vector<double>> gll_fields;

        if (has_partitions) {
            if (verbose)
                std::cout << "[model_to_vtk] Reading partitions/...\n";
            // Read from partition files
            // (simplified: just read from model.h5 if available)
            for (const auto& name : gll_field_names) {
                std::string path = "field/cell/" + name;
                if (dataset_exists(fm.id(), path)) {
                    std::vector<hsize_t> sh;
                    gll_fields[name] = read_float64_nd(fm.id(), path, sh);
                }
            }
        } else {
            if (verbose)
                std::cout << "[model_to_vtk] Reading fields from model.h5...\n";
            for (const auto& name : gll_field_names) {
                std::string path = "field/cell/" + name;
                if (dataset_exists(fm.id(), path)) {
                    std::vector<hsize_t> sh;
                    gll_fields[name] = read_float64_nd(fm.id(), path, sh);
                }
            }
        }

        // Read source coefficient from config.h5
        std::vector<double> source_coeff(n_cell * gll_per_cell, 0.0);
        if (dataset_exists(H5Fopen(config_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                           "source/elements/element_ids")) {
            H5File fc(config_path);
            if (dataset_exists(fc.id(), "source/elements/element_ids") &&
                dataset_exists(fc.id(), "source/elements/weights")) {
                auto eids = read_int64_1d(fc.id(), "source/elements/element_ids");
                std::vector<hsize_t> wsh;
                auto weights = read_float64_nd(fc.id(), "source/elements/weights", wsh);
                // wsh: [n_src, ngll, ngll, ngll]
                if (wsh.size() >= 4 && (int)wsh[1] == ngll) {
                    int64_t src_gll = (int64_t)ngll * ngll * ngll;
                    for (size_t si = 0; si < eids.size(); ++si) {
                        int64_t idx = eids[si] - 1;  // 1-based → 0-based
                        if (idx >= 0 && idx < n_cell) {
                            for (int64_t gp = 0; gp < src_gll; ++gp)
                                source_coeff[idx * gll_per_cell + gp] = weights[si * src_gll + gp];
                        }
                    }
                }
            }
        }

        // Assemble GLL point coordinates
        int64_t n_gll_pt = n_cell * gll_per_cell;
        std::vector<float> gll_points(n_gll_pt * 3, 0.0f);
#pragma omp parallel for collapse(3)
        for (int64_t ci = 0; ci < n_cell; ++ci) {
            for (int i = 0; i < ngll; ++i) {
                for (int j = 0; j < ngll; ++j) {
                    for (int k = 0; k < ngll; ++k) {
                        int64_t src_off =
                            ((ci * (int64_t)ngll + i) * (int64_t)ngll + j) * (int64_t)ngll + k;
                        int64_t dst_off = (ci * gll_per_cell + gll_idx(i, j, k, ngll)) * 3;
                        gll_points[dst_off + 0] = (float)gll_coords_flat[src_off * 3 + 0];
                        gll_points[dst_off + 1] = (float)gll_coords_flat[src_off * 3 + 1];
                        gll_points[dst_off + 2] = (float)gll_coords_flat[src_off * 3 + 2];
                    }
                }
            }
        }

        // Merge vertices
        vertex_coords_out.resize((n_vert + n_gll_pt) * 3);
        std::copy(vtx_coords.begin(), vtx_coords.end(), vertex_coords_out.begin());
        std::copy(gll_points.begin(), gll_points.end(), vertex_coords_out.begin() + n_vert * 3);

        VertexToCell vtc = build_vertex_to_cell(connectivity, n_cell, n_vert);

        // Build point fields
        if (verbose)
            std::cout << "[model_to_vtk] Building GLL point data and topology...\n";
        struct FieldSpec {
            const char* vtk_name;
            const char* raw_name;
        };
        FieldSpec fields[] = {
            {"Vp_m_s", "vp"}, {"Vs_m_s", "vs"},           {"Density_kg_m3", "density"},
            {"Mass", "mass"}, {"PML_Damping", "damping"}, {"Source_Coeff", "source_coeff"}};

        for (const auto& fs : fields) {
            std::string rname(fs.raw_name);
            auto it = gll_fields.find(rname);
            const std::vector<double>* raw_ptr = nullptr;
            if (it != gll_fields.end())
                raw_ptr = &it->second;

            std::vector<float> arr(vertex_coords_out.size() / 3, 0.0f);

            // Mesh vertex part: interpolate cell average
            if (raw_ptr) {
                std::vector<double> cell_avg(n_cell, 0.0);
#pragma omp parallel for
                for (int64_t ci = 0; ci < n_cell; ++ci) {
                    double sum = 0.0;
                    for (int64_t gp = 0; gp < gll_per_cell; ++gp)
                        sum += (*raw_ptr)[ci * gll_per_cell + gp];
                    cell_avg[ci] = sum / (double)gll_per_cell;
                }
                auto vert_interp = interpolate_mesh_vertex_field(cell_avg, vtc, n_vert);
                for (int64_t vi = 0; vi < n_vert; ++vi)
                    arr[vi] = vert_interp[vi];
            }

            // GLL point part: raw values
            if (rname == "source_coeff") {
#pragma omp parallel for
                for (int64_t ci = 0; ci < n_cell; ++ci) {
                    int64_t s = n_vert + ci * gll_per_cell;
                    for (int64_t gp = 0; gp < gll_per_cell; ++gp)
                        arr[s + gp] = (float)source_coeff[ci * gll_per_cell + gp];
                }
            } else if (raw_ptr) {
#pragma omp parallel for
                for (int64_t ci = 0; ci < n_cell; ++ci) {
                    int64_t s = n_vert + ci * gll_per_cell;
                    for (int64_t gp = 0; gp < gll_per_cell; ++gp)
                        arr[s + gp] = (float)(*raw_ptr)[ci * gll_per_cell + gp];
                }
            }

            point_fields[fs.vtk_name] = std::move(arr);
        }

        // Build GLL sub-cells
        gll_tmpl = build_gll_template(ngll);
        gll_cells = build_gll_cells(gll_tmpl, n_cell, ngll, n_vert);
        n_gll_cells = gll_cells.cell_types.size();

        if (verbose) {
            std::cout << "  Point fields: ";
            for (const auto& pf : point_fields)
                std::cout << pf.first << " ";
            std::cout << "\n  GLL per cell: " << gll_per_cell << ", total GLL: " << n_gll_pt
                      << "\n";
        }
    }

    // ── Write VTK ──────────────────────────────────────────────
    std::string out_path = vtk_dir + "/model.vtk";
    std::cout << "[model_to_vtk] Writing " << out_path << "\n";

    gf_vtk::VtkWriter vtk(out_path, "model.h5 converted to VTK (C++)");

    // Points
    vtk.write_points(vertex_coords_out);

    // Cells: hex + GLL sub-cells
    int64_t n_hex = n_cell;
    int64_t total_cells = n_hex + n_gll_cells;

    // Build hex cell array
    std::vector<int32_t> hex_cells(n_hex * 9);
#pragma omp parallel for
    for (int64_t ci = 0; ci < n_hex; ++ci) {
        int64_t off = ci * 9;
        hex_cells[off] = 8;
        for (int j = 0; j < 8; ++j)
            hex_cells[off + 1 + j] = (int32_t)connectivity[ci * 8 + j];
    }
    std::vector<int32_t> hex_types(n_hex, 12);  // VTK_HEXAHEDRON

    if (has_gll && ngll > 0) {
        // Combine: hex cells + GLL cells
        std::vector<int32_t> all_cells;
        all_cells.reserve(hex_cells.size() + gll_cells.cells.size());
        all_cells.insert(all_cells.end(), hex_cells.begin(), hex_cells.end());
        all_cells.insert(all_cells.end(), gll_cells.cells.begin(), gll_cells.cells.end());

        std::vector<int32_t> all_types;
        all_types.reserve(hex_types.size() + gll_cells.cell_types.size());
        all_types.insert(all_types.end(), hex_types.begin(), hex_types.end());
        all_types.insert(all_types.end(), gll_cells.cell_types.begin(),
                         gll_cells.cell_types.end());

        vtk.write_cells(all_cells, all_types);
    } else {
        vtk.write_cells(hex_cells, hex_types);
    }

    // Cell Data
    vtk.begin_cell_data(total_cells);
    for (auto& kv : cell_fields) {
        std::vector<float> padded(total_cells, 0.0f);
        std::copy(kv.second.begin(), kv.second.end(), padded.begin());
        // GLL cells get parent element value via elem_map
        if (has_gll && ngll > 0 && !gll_cells.elem_map.empty()) {
            for (size_t i = 0; i < gll_cells.elem_map.size(); ++i) {
                padded[n_hex + i] = kv.second[gll_cells.elem_map[i]];
            }
        }
        vtk.write_scalar_field(kv.first, padded);
    }

    // Point Data
    if (!point_fields.empty()) {
        vtk.begin_point_data(vertex_coords_out.size() / 3);
        for (auto& kv : point_fields) {
            vtk.write_scalar_field(kv.first, kv.second);
        }
    }

    std::cout << "  Done.\n";
    return 0;
}