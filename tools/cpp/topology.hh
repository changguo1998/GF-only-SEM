#ifndef GF_TOPOLOGY_HH
#define GF_TOPOLOGY_HH

#include <array>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// GMSH-like hexahedral topology resolvers and GLL sub-cell builders.
// All vertex indices are 0-based.

namespace gf_topology {

// Hex face vertex ordering (local indices 0-7)
static const int HEX_FACES[6][4] = {
    {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
    {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5}
};

// Resolve 8 global vertex indices for one hex element from topology tables.
// cell_surfaces[6] = signed surface IDs (1-based, sign = orientation)
// surface_edges[n_surface][4] = signed edge IDs
// edge_vertices[n_edge][2] = 1-based global vertex IDs
inline std::vector<int64_t> resolve_cell_vertices(
    const int64_t *cell_surfaces,
    const std::vector<std::vector<int64_t>> &surface_edges,
    const std::vector<std::vector<int64_t>> &edge_vertices)
{
    int64_t local_to_global[8];
    for (int i = 0; i < 8; ++i) local_to_global[i] = -1;

    for (int fi = 0; fi < 6; ++fi) {
        int64_t sid_signed = cell_surfaces[fi];
        int64_t sid = std::abs(sid_signed) - 1;
        const auto &canonical = surface_edges[sid];  // 4 signed edge IDs

        int64_t signed_edges[4];
        if (sid_signed > 0) {
            for (int k = 0; k < 4; ++k) signed_edges[k] = canonical[k];
        } else {
            // Reverse face orientation
            signed_edges[0] = -canonical[3];
            signed_edges[1] = -canonical[2];
            signed_edges[2] = -canonical[1];
            signed_edges[3] = -canonical[0];
        }

        for (int k = 0; k < 4; ++k) {
            int64_t eid = std::abs(signed_edges[k]) - 1;
            int64_t gv1 = edge_vertices[eid][0] - 1;  // 1-based → 0-based
            int64_t gv2 = edge_vertices[eid][1] - 1;
            int lvk = HEX_FACES[fi][k];
            int lvk_next = HEX_FACES[fi][(k + 1) % 4];
            if (signed_edges[k] > 0) {
                local_to_global[lvk] = gv1;
                local_to_global[lvk_next] = gv2;
            } else {
                local_to_global[lvk] = gv2;
                local_to_global[lvk_next] = gv1;
            }
        }
    }

    std::vector<int64_t> result(8);
    for (int i = 0; i < 8; ++i) result[i] = local_to_global[i];
    return result;
}

// Inline flat GLL index
inline int64_t gll_idx(int i, int j, int k, int ngll) {
    return (int64_t)i * ngll * ngll + j * ngll + k;
}

// Precompute GLL sub-cell templates for given polynomial order.
struct GllTemplate {
    std::vector<std::pair<int64_t, int64_t>> edge_lines; // (a, b) local GLL indices
    std::vector<std::array<int64_t, 4>> face_quads;      // 4 local GLL indices
    std::vector<std::array<int64_t, 8>> sub_hexes;       // 8 local GLL indices
};

inline GllTemplate build_gll_template(int ngll) {
    GllTemplate t;
    auto idx = [ngll](int i, int j, int k) { return gll_idx(i, j, k, ngll); };

    // 12 edges × (ngll-1) LINEs
    // Edge 0-1: along i, j=0, k=0
    for (int i = 0; i < ngll - 1; ++i) t.edge_lines.push_back({idx(i,0,0), idx(i+1,0,0)});
    // Edge 2-3: along i, j=ngll-1, k=0
    for (int i = 0; i < ngll - 1; ++i) t.edge_lines.push_back({idx(i,ngll-1,0), idx(i+1,ngll-1,0)});
    // Edge 4-5: along i, j=0, k=ngll-1
    for (int i = 0; i < ngll - 1; ++i) t.edge_lines.push_back({idx(i,0,ngll-1), idx(i+1,0,ngll-1)});
    // Edge 6-7: along i, j=ngll-1, k=ngll-1
    for (int i = 0; i < ngll - 1; ++i) t.edge_lines.push_back({idx(i,ngll-1,ngll-1), idx(i+1,ngll-1,ngll-1)});
    // Edge 0-2: along j, i=0, k=0
    for (int j = 0; j < ngll - 1; ++j) t.edge_lines.push_back({idx(0,j,0), idx(0,j+1,0)});
    // Edge 1-3: along j, i=ngll-1, k=0
    for (int j = 0; j < ngll - 1; ++j) t.edge_lines.push_back({idx(ngll-1,j,0), idx(ngll-1,j+1,0)});
    // Edge 4-6: along j, i=0, k=ngll-1
    for (int j = 0; j < ngll - 1; ++j) t.edge_lines.push_back({idx(0,j,ngll-1), idx(0,j+1,ngll-1)});
    // Edge 5-7: along j, i=ngll-1, k=ngll-1
    for (int j = 0; j < ngll - 1; ++j) t.edge_lines.push_back({idx(ngll-1,j,ngll-1), idx(ngll-1,j+1,ngll-1)});
    // Edge 0-4: along k, i=0, j=0
    for (int k = 0; k < ngll - 1; ++k) t.edge_lines.push_back({idx(0,0,k), idx(0,0,k+1)});
    // Edge 1-5: along k, i=ngll-1, j=0
    for (int k = 0; k < ngll - 1; ++k) t.edge_lines.push_back({idx(ngll-1,0,k), idx(ngll-1,0,k+1)});
    // Edge 3-7: along k, i=0, j=ngll-1
    for (int k = 0; k < ngll - 1; ++k) t.edge_lines.push_back({idx(0,ngll-1,k), idx(0,ngll-1,k+1)});
    // Edge 2-6: along k, i=ngll-1, j=ngll-1
    for (int k = 0; k < ngll - 1; ++k) t.edge_lines.push_back({idx(ngll-1,ngll-1,k), idx(ngll-1,ngll-1,k+1)});

    // 6 faces × (ngll-1)² QUADs
    // Face 0 (z=0): along i,j
    for (int i = 0; i < ngll - 1; ++i)
        for (int j = 0; j < ngll - 1; ++j)
            t.face_quads.push_back({idx(i,j,0), idx(i+1,j,0), idx(i+1,j+1,0), idx(i,j+1,0)});
    // Face 1 (z=ngll-1): along i,j
    for (int i = 0; i < ngll - 1; ++i)
        for (int j = 0; j < ngll - 1; ++j)
            t.face_quads.push_back({idx(i,j,ngll-1), idx(i+1,j,ngll-1), idx(i+1,j+1,ngll-1), idx(i,j+1,ngll-1)});
    // Face 2 (y=0): along i,k
    for (int i = 0; i < ngll - 1; ++i)
        for (int k = 0; k < ngll - 1; ++k)
            t.face_quads.push_back({idx(i,0,k), idx(i+1,0,k), idx(i+1,0,k+1), idx(i,0,k+1)});
    // Face 3 (y=ngll-1): along i,k
    for (int i = 0; i < ngll - 1; ++i)
        for (int k = 0; k < ngll - 1; ++k)
            t.face_quads.push_back({idx(i,ngll-1,k), idx(i+1,ngll-1,k), idx(i+1,ngll-1,k+1), idx(i,ngll-1,k+1)});
    // Face 4 (x=0): along j,k
    for (int j = 0; j < ngll - 1; ++j)
        for (int k = 0; k < ngll - 1; ++k)
            t.face_quads.push_back({idx(0,j,k), idx(0,j+1,k), idx(0,j+1,k+1), idx(0,j,k+1)});
    // Face 5 (x=ngll-1): along j,k
    for (int j = 0; j < ngll - 1; ++j)
        for (int k = 0; k < ngll - 1; ++k)
            t.face_quads.push_back({idx(ngll-1,j,k), idx(ngll-1,j+1,k), idx(ngll-1,j+1,k+1), idx(ngll-1,j,k+1)});

    // (ngll-1)³ sub-volume HEXs
    for (int i = 0; i < ngll - 1; ++i)
        for (int j = 0; j < ngll - 1; ++j)
            for (int k = 0; k < ngll - 1; ++k)
                t.sub_hexes.push_back({
                    idx(i,j,k), idx(i+1,j,k), idx(i+1,j+1,k), idx(i,j+1,k),
                    idx(i,j,k+1), idx(i+1,j,k+1), idx(i+1,j+1,k+1), idx(i,j+1,k+1)
                });

    return t;
}

// Build cell arrays for VTK from GLL template.
// Returns (cells_flat, cell_types, elem_map).
struct GllCells {
    std::vector<int32_t> cells;      // flattened CELLS array
    std::vector<int32_t> cell_types; // per-cell VTK type
    std::vector<int32_t> elem_map;   // parent element index for each GLL cell
};

inline GllCells build_gll_cells(
    const GllTemplate &tmpl, int64_t n_local, int ngll, int64_t n_mesh_vert)
{
    GllCells out;
    int64_t gll_per_cell = (int64_t)ngll * ngll * ngll;
    int64_t n_edge = n_local * tmpl.edge_lines.size();
    int64_t n_face = n_local * tmpl.face_quads.size();
    int64_t n_sub  = n_local * tmpl.sub_hexes.size();

    // Pre-allocate
    out.cells.reserve(n_edge * 3 + n_face * 5 + n_sub * 9);
    out.cell_types.reserve(n_edge + n_face + n_sub);
    out.elem_map.reserve(n_edge + n_face + n_sub);

    for (int64_t li = 0; li < n_local; ++li) {
        int64_t base = n_mesh_vert + li * gll_per_cell;

        for (const auto &e : tmpl.edge_lines) {
            out.cells.push_back(2);
            out.cells.push_back((int32_t)(base + e.first));
            out.cells.push_back((int32_t)(base + e.second));
            out.cell_types.push_back(3); // VTK_LINE
            out.elem_map.push_back((int32_t)li);
        }
        for (const auto &f : tmpl.face_quads) {
            out.cells.push_back(4);
            for (int i = 0; i < 4; ++i)
                out.cells.push_back((int32_t)(base + f[i]));
            out.cell_types.push_back(9); // VTK_QUAD
            out.elem_map.push_back((int32_t)li);
        }
        for (const auto &h : tmpl.sub_hexes) {
            out.cells.push_back(8);
            for (int i = 0; i < 8; ++i)
                out.cells.push_back((int32_t)(base + h[i]));
            out.cell_types.push_back(12); // VTK_HEXAHEDRON
            out.elem_map.push_back((int32_t)li);
        }
    }
    return out;
}

} // namespace gf_topology

#endif // GF_TOPOLOGY_HH