/// metis_partition.cpp — METIS C API wrapper + topology-driven global node numbering
///
/// Replaces the pymetis Python wrapper.  Calls the METIS C API directly for
/// k-way partition, then assigns global node IDs using topology face adjacency
/// (ported from preprocess/partition.py).

#include <hdf5.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include "gf_config.h"
#include "gf_hdf5_helpers.hpp"

// METIS C API (spack-installed)
extern "C" {
#include <metis.h>
}

namespace gf {
namespace {

// ── Face GLL node ordering ─────────────────────────────────────────────────

/// Face GLL node indices (flattened, 0-based) for a given face.
/// Face 0=-z, 1=+z, 2=-y, 3=+y, 4=-x, 5=+x.
std::vector<int> face_gll_nodes(int face_index, int ngll) {
    std::vector<int> nodes;
    nodes.reserve(ngll * ngll);
    switch (face_index) {
        case 0:  // -z: k=0
            for (int i = 0; i < ngll; ++i)
                for (int j = 0; j < ngll; ++j)
                    nodes.push_back((i * ngll + j) * ngll + 0);
            break;
        case 1:  // +z: k=ngll-1
            for (int i = 0; i < ngll; ++i)
                for (int j = 0; j < ngll; ++j)
                    nodes.push_back((i * ngll + j) * ngll + (ngll - 1));
            break;
        case 2:  // -y: j=0
            for (int i = 0; i < ngll; ++i)
                for (int k = 0; k < ngll; ++k)
                    nodes.push_back((i * ngll + 0) * ngll + k);
            break;
        case 3:  // +y: j=ngll-1
            for (int i = 0; i < ngll; ++i)
                for (int k = 0; k < ngll; ++k)
                    nodes.push_back((i * ngll + (ngll - 1)) * ngll + k);
            break;
        case 4:  // -x: i=0
            for (int j = 0; j < ngll; ++j)
                for (int k = 0; k < ngll; ++k)
                    nodes.push_back((0 * ngll + j) * ngll + k);
            break;
        case 5:  // +x: i=ngll-1
            for (int j = 0; j < ngll; ++j)
                for (int k = 0; k < ngll; ++k)
                    nodes.push_back(((ngll - 1) * ngll + j) * ngll + k);
            break;
    }
    return nodes;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════════
//  METIS partition (k-way)
// ═════════════════════════════════════════════════════════════════════════════

void partition_metis(const char* model_path, int n_ranks) {
    fprintf(stderr, "=== METIS partition ===\n");

    hid_t model_fid = h5::open_or_fail(model_path, H5F_ACC_RDWR);

    // Read cell_to_surface topology
    std::vector<int64_t> cell_to_surface = h5::read_int64(model_fid, "topology/cell_to_surface");
    std::vector<hsize_t> c2s_dims = h5::get_dims(model_fid, "topology/cell_to_surface");
    int n_cell = static_cast<int>(c2s_dims[0]);
    int n_faces_per_cell = static_cast<int>(c2s_dims[1]);

    // Single rank: trivial partition
    if (n_ranks <= 1 || n_cell <= 1) {
        std::vector<int32_t> element_to_rank(n_cell, 0);
        h5::write_int32(model_fid, "partition/element_to_rank", element_to_rank,
                        {static_cast<hsize_t>(n_cell)});
        H5Fclose(model_fid);
        fprintf(stderr, "  Single rank — all %d elements on rank 0\n", n_cell);
        return;
    }

    // Build adjacency list from cell_to_surface.
    // cell_to_surface[e][face] = signed surface id (1-based, sign = orientation).
    // Two cells sharing the same |surface_id| are face-neighbours.

    std::map<int64_t, std::vector<int>> surface_to_cells;
    for (int e = 0; e < n_cell; ++e) {
        for (int f = 0; f < n_faces_per_cell; ++f) {
            int64_t sid = cell_to_surface[e * n_faces_per_cell + f];
            if (sid != 0)
                surface_to_cells[std::abs(sid)].push_back(e);
        }
    }

    std::vector<std::vector<int>> adjacency(n_cell);
    for (const auto& [sid, cells] : surface_to_cells) {
        for (size_t i = 0; i < cells.size(); ++i) {
            for (size_t j = i + 1; j < cells.size(); ++j) {
                int c1 = cells[i], c2 = cells[j];
                if (std::find(adjacency[c1].begin(), adjacency[c1].end(), c2) ==
                    adjacency[c1].end())
                    adjacency[c1].push_back(c2);
                if (std::find(adjacency[c2].begin(), adjacency[c2].end(), c1) ==
                    adjacency[c2].end())
                    adjacency[c2].push_back(c1);
            }
        }
    }

    // Convert to CSR format required by METIS
    std::vector<idx_t> xadj(n_cell + 1, 0);
    std::vector<idx_t> adjncy;
    for (int e = 0; e < n_cell; ++e) {
        xadj[e + 1] = xadj[e] + static_cast<idx_t>(adjacency[e].size());
        for (int nb : adjacency[e])
            adjncy.push_back(static_cast<idx_t>(nb));
    }

    // Call METIS k-way partition
    idx_t nvtxs = static_cast<idx_t>(n_cell);
    idx_t ncon = 1;
    idx_t nparts = static_cast<idx_t>(n_ranks);
    idx_t edgecut = 0;
    // Allocate METIS partition result array (one element→rank mapping per cell).
    std::vector<idx_t> metis_partition(n_cell, 0);

    int metis_rc = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj.data(), adjncy.data(),
                                            /*vwgt=*/nullptr, /*vsize=*/nullptr,
                                            /*adjwgt=*/nullptr,  // no weights
                                            &nparts, /*tpwgts=*/nullptr, /*ubvec=*/nullptr,
                                            /*options=*/nullptr, &edgecut, metis_partition.data());

    if (metis_rc != METIS_OK) {
        fprintf(stderr, "ERROR: METIS_PartGraphRecursive failed with code %d\n", metis_rc);
        std::exit(1);
    }

    fprintf(stderr, "  METIS: %d elements → %d ranks, edgecut=%d\n", n_cell, n_ranks,
            static_cast<int>(edgecut));

    // Write element_to_rank (METIS returns 0-based partition)
    std::vector<int32_t> element_to_rank(n_cell);
    for (int e = 0; e < n_cell; ++e)
        element_to_rank[e] = static_cast<int32_t>(metis_partition[e]);

    h5::write_int32(model_fid, "partition/element_to_rank", element_to_rank,
                    {static_cast<hsize_t>(n_cell)});

    // Write n_ranks attribute on the dataset
    hid_t ds_etr = H5Dopen2(model_fid, "partition/element_to_rank", H5P_DEFAULT);
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t attr =
        H5Acreate2(ds_etr, "n_ranks", H5T_NATIVE_INT32, attr_space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT32, &n_ranks);
    H5Aclose(attr);
    H5Sclose(attr_space);
    H5Dclose(ds_etr);

    H5Fclose(model_fid);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Topology-driven global node numbering
// ═════════════════════════════════════════════════════════════════════════════

void compute_global_node_ids(const char* model_path, int ngll) {
    fprintf(stderr, "=== Global node numbering (topology) ===\n");

    hid_t model_fid = h5::open_or_fail(model_path, H5F_ACC_RDWR);

    // Read cell_to_surface and determine array dimensions
    std::vector<int64_t> cell_to_surface = h5::read_int64(model_fid, "topology/cell_to_surface");
    std::vector<hsize_t> c2s_dims = h5::get_dims(model_fid, "topology/cell_to_surface");
    int n_cell = static_cast<int>(c2s_dims[0]);
    int n_faces = static_cast<int>(c2s_dims[1]);
    int n_node_per_cell = ngll * ngll * ngll;

    // Precompute face GLL node index lists for all 6 faces
    std::vector<std::vector<int>> face_nodes(6);
    for (int f = 0; f < 6; ++f)
        face_nodes[f] = face_gll_nodes(f, ngll);

    // Build surface → [(cell, face)] map for shared-face detection
    std::map<int64_t, std::vector<std::pair<int, int>>> surface_to_cellface;
    std::vector<std::map<int64_t, int>> element_surface_to_face(n_cell);

    for (int e = 0; e < n_cell; ++e) {
        for (int f = 0; f < n_faces; ++f) {
            int64_t sid = cell_to_surface[e * n_faces + f];
            if (sid != 0) {
                int64_t abs_sid = std::abs(sid);
                surface_to_cellface[abs_sid].emplace_back(e, f);
                element_surface_to_face[e][abs_sid] = f;
            }
        }
    }

    // Initialize global IDs to -1 (unassigned)
    std::vector<int32_t> global_node_ids(n_cell * n_node_per_cell, -1);
    int next_global_id = 0;

    // Pass 1: match shared-face nodes between face-neighbour elements
    for (const auto& [sid, cellfaces] : surface_to_cellface) {
        if (cellfaces.size() != 2)
            continue;  // skip boundary surfaces

        auto [e0, f0] = cellfaces[0];
        auto [e1, f1] = cellfaces[1];

        const auto& nodes_f0 = face_nodes[f0];
        const auto& nodes_f1 = face_nodes[f1];

        for (size_t k = 0; k < nodes_f0.size(); ++k) {
            int flat0 = e0 * n_node_per_cell + nodes_f0[k];
            int flat1 = e1 * n_node_per_cell + nodes_f1[k];

            int gid0 = global_node_ids[flat0];
            int gid1 = global_node_ids[flat1];

            if (gid0 < 0 && gid1 < 0) {
                global_node_ids[flat0] = next_global_id;
                global_node_ids[flat1] = next_global_id;
                ++next_global_id;
            } else if (gid0 >= 0 && gid1 < 0) {
                global_node_ids[flat1] = gid0;
            } else if (gid0 < 0 && gid1 >= 0) {
                global_node_ids[flat0] = gid1;
            }
        }
    }

    // Pass 2: assign remaining unassigned (interior) nodes
    for (int flat = 0; flat < n_cell * n_node_per_cell; ++flat) {
        if (global_node_ids[flat] < 0)
            global_node_ids[flat] = next_global_id++;
    }

    fprintf(stderr, "  Global nodes: %d (n_cell=%d, ngll=%d)\n", next_global_id, n_cell, ngll);

    // Write to HDF5
    std::vector<hsize_t> gnode_dims = {static_cast<hsize_t>(n_cell), static_cast<hsize_t>(ngll),
                                       static_cast<hsize_t>(ngll), static_cast<hsize_t>(ngll)};
    h5::write_int32(model_fid, "partition/global_cell2global_node", global_node_ids, gnode_dims);

    H5Fclose(model_fid);
}

}  // namespace gf