/// metis_partition.cpp — METIS C API wrapper + topology-driven global node numbering
///
/// Replaces pymetis Python wrapper.  Calls METIS C API directly for k-way
/// partition, then assigns global node IDs using topology face adjacency
/// (ported from Python partition.py).

#include <hdf5.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#include "gf_config.h"

// METIS C API (spack-installed)
extern "C" {
#include <metis.h>
}

namespace gf {
namespace {

// ── Helpers ──────────────────────────────────────────────────────────────

/// Open HDF5 file, exit on failure.
hid_t open_or_fail(const char* path, unsigned flags) {
    hid_t fid = H5Fopen(path, flags, H5P_DEFAULT);
    if (fid < 0) {
        fprintf(stderr, "ERROR: cannot open HDF5 file: %s\n", path);
        std::exit(1);
    }
    return fid;
}

/// Read 1-D int64 dataset.
std::vector<int64_t> read_int64(hid_t fid, const char* name) {
    hid_t ds = H5Dopen2(fid, name, H5P_DEFAULT);
    if (ds < 0) {
        fprintf(stderr, "ERROR: dataset not found: %s\n", name);
        std::exit(1);
    }
    hid_t space = H5Dget_space(ds);
    hsize_t dims[8];
    int ndims = H5Sget_simple_extent_ndims(space);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    hsize_t total = 1;
    for (int i = 0; i < ndims; ++i)
        total *= dims[i];
    std::vector<int64_t> buf(total);
    H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(space);
    return buf;
}

/// Write int32 dataset.
void write_int32(hid_t fid, const char* name, const std::vector<int32_t>& data,
                 const std::vector<hsize_t>& dims) {
    H5Ldelete(fid, name, H5P_DEFAULT);
    hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
    hid_t ds =
        H5Dcreate2(fid, name, H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(ds);
    H5Sclose(space);
}

/// Write int64 dataset.
void write_int64(hid_t fid, const char* name, const std::vector<int64_t>& data,
                 const std::vector<hsize_t>& dims) {
    H5Ldelete(fid, name, H5P_DEFAULT);
    hid_t space = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
    hid_t ds =
        H5Dcreate2(fid, name, H5T_NATIVE_INT64, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Dclose(ds);
    H5Sclose(space);
}

/// Face GLL node indices (flattened, 0-based).
std::vector<int> face_gll_nodes(int face_idx, int ngll) {
    std::vector<int> nodes;
    nodes.reserve(ngll * ngll);
    if (face_idx == 0) {  // -z: k=0
        for (int i = 0; i < ngll; ++i)
            for (int j = 0; j < ngll; ++j)
                nodes.push_back((i * ngll + j) * ngll + 0);
    } else if (face_idx == 1) {  // +z: k=ngll-1
        for (int i = 0; i < ngll; ++i)
            for (int j = 0; j < ngll; ++j)
                nodes.push_back((i * ngll + j) * ngll + (ngll - 1));
    } else if (face_idx == 2) {  // -y: j=0
        for (int i = 0; i < ngll; ++i)
            for (int k = 0; k < ngll; ++k)
                nodes.push_back((i * ngll + 0) * ngll + k);
    } else if (face_idx == 3) {  // +y: j=ngll-1
        for (int i = 0; i < ngll; ++i)
            for (int k = 0; k < ngll; ++k)
                nodes.push_back((i * ngll + (ngll - 1)) * ngll + k);
    } else if (face_idx == 4) {  // -x: i=0
        for (int j = 0; j < ngll; ++j)
            for (int k = 0; k < ngll; ++k)
                nodes.push_back((0 * ngll + j) * ngll + k);
    } else if (face_idx == 5) {  // +x: i=ngll-1
        for (int j = 0; j < ngll; ++j)
            for (int k = 0; k < ngll; ++k)
                nodes.push_back(((ngll - 1) * ngll + j) * ngll + k);
    }
    return nodes;
}

}  // namespace

// ── METIS partition ─────────────────────────────────────────────────────

void partition_metis(const char* model_path, int n_ranks) {
    fprintf(stderr, "=== METIS partition ===\n");

    hid_t fid = open_or_fail(model_path, H5F_ACC_RDWR);

    // Read cell_to_surface
    hid_t ds = H5Dopen2(fid, "topology/cell_to_surface", H5P_DEFAULT);
    hid_t sp = H5Dget_space(ds);
    hsize_t cdims[8];
    int cnd = H5Sget_simple_extent_ndims(sp);
    H5Sget_simple_extent_dims(sp, cdims, nullptr);
    int n_cell = static_cast<int>(cdims[0]);
    int n_faces_per_cell = static_cast<int>(cdims[1]);  // should be 6
    std::vector<int64_t> c2s_flat(n_cell * n_faces_per_cell);
    H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, c2s_flat.data());
    H5Dclose(ds);
    H5Sclose(sp);

    if (n_ranks <= 1 || n_cell <= 1) {
        // Single rank: all elements to rank 0
        std::vector<int32_t> element_to_rank(n_cell, 0);
        write_int32(fid, "partition/element_to_rank", element_to_rank,
                    {static_cast<hsize_t>(n_cell)});
        H5Fclose(fid);
        fprintf(stderr, "  Single rank — all %d elements on rank 0\n", n_cell);
        return;
    }

    // Build adjacency list from cell_to_surface
    // cell_to_surface[e][face] = signed surface id (1-based, positive/negative for orientation)
    // Build surface → cells map
    std::map<int64_t, std::vector<int>> surf_cells;
    for (int e = 0; e < n_cell; ++e) {
        for (int f = 0; f < n_faces_per_cell; ++f) {
            int64_t sid = c2s_flat[e * n_faces_per_cell + f];
            if (sid != 0) {
                int64_t abs_sid = std::abs(sid);
                surf_cells[abs_sid].push_back(e);
            }
        }
    }

    // Build adjacency
    std::vector<std::vector<int>> adj(n_cell);
    for (auto& [sid, cells] : surf_cells) {
        for (size_t i = 0; i < cells.size(); ++i) {
            for (size_t j = i + 1; j < cells.size(); ++j) {
                int c1 = cells[i], c2 = cells[j];
                if (std::find(adj[c1].begin(), adj[c1].end(), c2) == adj[c1].end())
                    adj[c1].push_back(c2);
                if (std::find(adj[c2].begin(), adj[c2].end(), c1) == adj[c2].end())
                    adj[c2].push_back(c1);
            }
        }
    }

    // Convert to CSR format for METIS
    std::vector<idx_t> xadj(n_cell + 1, 0);
    std::vector<idx_t> adjncy;
    for (int e = 0; e < n_cell; ++e) {
        xadj[e + 1] = xadj[e] + static_cast<idx_t>(adj[e].size());
        for (int nb : adj[e])
            adjncy.push_back(static_cast<idx_t>(nb));
    }

    // METIS options
    idx_t nvtxs = static_cast<idx_t>(n_cell);
    idx_t ncon = 1;
    idx_t nparts = static_cast<idx_t>(n_ranks);
    idx_t edgecut = 0;
    std::vector<idx_t> part(n_cell, 0);

    int metis_rc = METIS_PartGraphRecursive(
        &nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr,  // no weights
        &nparts, nullptr, nullptr, nullptr, &edgecut, part.data());

    if (metis_rc != METIS_OK) {
        fprintf(stderr, "ERROR: METIS_PartGraphRecursive failed with code %d\n", metis_rc);
        std::exit(1);
    }

    fprintf(stderr, "  METIS: %d elements → %d ranks, edgecut=%d\n", n_cell, n_ranks,
            (int)edgecut);

    // Write element_to_rank (METIS returns 0-based partition)
    std::vector<int32_t> element_to_rank(n_cell);
    for (int e = 0; e < n_cell; ++e)
        element_to_rank[e] = static_cast<int32_t>(part[e]);

    H5Ldelete(fid, "partition", H5P_DEFAULT);
    hid_t part_grp = H5Gcreate2(fid, "partition", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(part_grp);

    write_int32(fid, "partition/element_to_rank", element_to_rank, {static_cast<hsize_t>(n_cell)});

    // Write n_ranks as attribute
    hid_t ds_etr = H5Dopen2(fid, "partition/element_to_rank", H5P_DEFAULT);
    hid_t attr_sp = H5Screate(H5S_SCALAR);
    hid_t attr =
        H5Acreate2(ds_etr, "n_ranks", H5T_NATIVE_INT32, attr_sp, H5P_DEFAULT, H5P_DEFAULT);
    int n_ranks_int = n_ranks;
    H5Awrite(attr, H5T_NATIVE_INT32, &n_ranks_int);
    H5Aclose(attr);
    H5Sclose(attr_sp);
    H5Dclose(ds_etr);

    H5Fclose(fid);
}

// ── Topology-driven global node numbering ───────────────────────────────

void compute_global_node_ids(const char* model_path, int ngll) {
    fprintf(stderr, "=== Global node numbering (topology) ===\n");

    hid_t fid = open_or_fail(model_path, H5F_ACC_RDWR);

    // Read cell_to_surface + get n_cell
    hid_t ds = H5Dopen2(fid, "topology/cell_to_surface", H5P_DEFAULT);
    hid_t sp = H5Dget_space(ds);
    hsize_t cdims[8];
    int cnd = H5Sget_simple_extent_ndims(sp);
    H5Sget_simple_extent_dims(sp, cdims, nullptr);
    int n_cell = static_cast<int>(cdims[0]);
    int n_faces = static_cast<int>(cdims[1]);
    std::vector<int64_t> c2s_flat(n_cell * n_faces);
    H5Dread(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, c2s_flat.data());
    H5Dclose(ds);
    H5Sclose(sp);

    int n_node = ngll * ngll * ngll;

    // Initialize global IDs to -1
    std::vector<int32_t> global_ids(n_cell * n_node, -1);

    // Precompute face GLL node lists
    std::vector<std::vector<int>> face_nodes(6);
    for (int f = 0; f < 6; ++f)
        face_nodes[f] = face_gll_nodes(f, ngll);

    // Build surface → [(cell, face)] map
    std::map<int64_t, std::vector<std::pair<int, int>>> surf_cellface;
    std::vector<std::map<int64_t, int>> elem_surf_face(n_cell);

    for (int e = 0; e < n_cell; ++e) {
        for (int f = 0; f < n_faces; ++f) {
            int64_t sid = c2s_flat[e * n_faces + f];
            if (sid != 0) {
                int64_t abs_sid = std::abs(sid);
                surf_cellface[abs_sid].push_back({e, f});
                elem_surf_face[e][abs_sid] = f;
            }
        }
    }

    int next_id = 0;

    // Pass 1: match shared-face nodes
    for (auto& [sid, cellfaces] : surf_cellface) {
        if (cellfaces.size() != 2)
            continue;

        auto [e0, f0] = cellfaces[0];
        auto [e1, f1] = cellfaces[1];

        auto& n0 = face_nodes[f0];
        auto& n1 = face_nodes[f1];

        for (size_t k = 0; k < n0.size(); ++k) {
            int flat0 = e0 * n_node + n0[k];
            int flat1 = e1 * n_node + n1[k];

            int gid0 = global_ids[flat0];
            int gid1 = global_ids[flat1];

            if (gid0 < 0 && gid1 < 0) {
                global_ids[flat0] = next_id;
                global_ids[flat1] = next_id;
                ++next_id;
            } else if (gid0 >= 0 && gid1 < 0) {
                global_ids[flat1] = gid0;
            } else if (gid0 < 0 && gid1 >= 0) {
                global_ids[flat0] = gid1;
            }
        }
    }

    // Pass 2: assign remaining unassigned nodes
    for (int flat = 0; flat < n_cell * n_node; ++flat) {
        if (global_ids[flat] < 0) {
            global_ids[flat] = next_id++;
        }
    }

    fprintf(stderr, "  Global nodes: %d (n_cell=%d, ngll=%d)\n", next_id, n_cell, ngll);

    // Write to HDF5
    H5Ldelete(fid, "partition/global_cell2global_node", H5P_DEFAULT);
    std::vector<hsize_t> gdims = {static_cast<hsize_t>(n_cell), static_cast<hsize_t>(ngll),
                                  static_cast<hsize_t>(ngll), static_cast<hsize_t>(ngll)};
    write_int32(fid, "partition/global_cell2global_node", global_ids, gdims);

    H5Fclose(fid);
}

}  // namespace gf