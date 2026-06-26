#!/usr/bin/env python3
"""Convert GMSH v4.1 .msh to internal HDF5 topology format.

Reads GMSH mesh, extracts topology only (vertices, edges, surfaces, cells),
writes mesh.h5 with /topology/ group. 1-based indexing, signed direction.

Usage:
    python tools/gmsh_to_hdf5.py input.msh [-o mesh.h5]
"""

import argparse
from collections import defaultdict
import h5py
import meshio
import numpy as np

# Hex face definitions (local vertex indices, CCW from outside)
# GMSH hex ordering:
#   v0(0,0,0) v1(1,0,0) v2(1,1,0) v3(0,1,0)  (bottom)
#   v4(0,0,1) v5(1,0,1) v6(1,1,1) v7(0,1,1)  (top)
HEX_FACES = [
    [0, 3, 2, 1],  # -z (bottom)
    [4, 5, 6, 7],  # +z (top)
    [0, 1, 5, 4],  # -y (front)
    [3, 7, 6, 2],  # +y (back)
    [0, 4, 7, 3],  # -x (left)
    [1, 2, 6, 5],  # +x (right)
]


def _same_orientation(sa, sb):
    """Check if two signed edge loops have same cyclic orientation (up to rotation)."""
    n = len(sa)
    first_abs = abs(sa[0])
    for offset in range(n):
        if abs(sb[offset]) == first_abs:
            rotated_b = [sb[(offset + i) % n] for i in range(n)]
            if all(sa[i] == rotated_b[i] for i in range(n)):
                return True
            break
    return False


def extract_topology(mesh):
    """Extract vertices, edges, surfaces, cells from GMSH mesh.

    Returns dict with:
      vertex_to_coord: (n_vertex, 3) float64
      edge_to_vertex:  (n_edge, 2) int64 — (+v1, +v2), v1 < v2
      surface_to_edge: (n_surface, 4) int64 — signed edge ids, CCW
      cell_to_surface: (n_cell, 6) int64 — signed surface ids
    """
    hex_cells = mesh.cells_dict.get("hexahedron")
    if hex_cells is None or hex_cells.shape[0] == 0:
        raise ValueError("No hexahedron cells found in mesh")

    n_cell = hex_cells.shape[0]
    n_vertex = mesh.points.shape[0]
    vertex_to_coord = mesh.points.astype(np.float64)

    # ── Edge deduplication ─────────────────────────────────────────────
    # key = (v_low, v_high) -> edge_id (1-based)
    edge_key_to_id = {}
    edge_pairs = []

    per_cell_edge_ids = []  # [ncells][12]  local-edge-index -> global edge id
    for cell_hex in hex_cells:
        v = cell_hex
        local_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        eids = []
        for a, b in local_edges:
            va, vb = int(v[a]), int(v[b])
            key = (va, vb) if va < vb else (vb, va)
            if key not in edge_key_to_id:
                edge_key_to_id[key] = len(edge_pairs) + 1
                edge_pairs.append(key)
            eids.append(edge_key_to_id[key])
        per_cell_edge_ids.append(eids)

    n_edge = len(edge_pairs)
    edge_to_vertex = np.zeros((n_edge, 2), dtype=np.int64)
    for eid, (vl, vh) in enumerate(edge_pairs):
        edge_to_vertex[eid] = [vl + 1, vh + 1]

    # ── Surface deduplication ──────────────────────────────────────────
    surf_unsigned_to_id = {}
    surf_edges_list = []
    cell_to_surface = np.zeros((n_cell, 6), dtype=np.int64)

    for ci, cell_hex in enumerate(hex_cells):
        v = cell_hex
        local_eids = per_cell_edge_ids[ci]

        for fi, face_verts in enumerate(HEX_FACES):
            signed_edges = []
            for k in range(4):
                va = int(v[face_verts[k]])
                vb = int(v[face_verts[(k + 1) % 4]])
                key = (va, vb) if va < vb else (vb, va)
                eid = edge_key_to_id[key]
                if va == key[0] and vb == key[1]:
                    signed_edges.append(eid)
                else:
                    signed_edges.append(-eid)

            unsigned = frozenset(abs(e) for e in signed_edges)
            if unsigned not in surf_unsigned_to_id:
                surf_unsigned_to_id[unsigned] = len(surf_edges_list) + 1
                surf_edges_list.append(list(signed_edges))

            sid = surf_unsigned_to_id[unsigned]
            canonical = surf_edges_list[sid - 1]
            if _same_orientation(signed_edges, canonical):
                cell_to_surface[ci, fi] = sid
            else:
                cell_to_surface[ci, fi] = -sid

    n_surface = len(surf_edges_list)
    surface_to_edge = np.zeros((n_surface, 4), dtype=np.int64)
    for sid in range(n_surface):
        surface_to_edge[sid] = surf_edges_list[sid]

    return {
        "vertex_to_coord": vertex_to_coord,
        "edge_to_vertex": edge_to_vertex,
        "surface_to_edge": surface_to_edge,
        "cell_to_surface": cell_to_surface,
    }


def write_topology(path, topology):
    """Write mesh.h5 with /topology/ group."""
    with h5py.File(path, "w") as f:
        topo = f.create_group("topology")
        v2c = topology["vertex_to_coord"]
        e2v = topology["edge_to_vertex"]
        s2e = topology["surface_to_edge"]
        c2s = topology["cell_to_surface"]

        n_vertex = v2c.shape[0]
        n_edge = e2v.shape[0]
        n_surface = s2e.shape[0]
        n_cell = c2s.shape[0]

        topo.create_dataset("vertex_to_coord", data=v2c, dtype="float64")
        topo.create_dataset("edge_to_vertex", data=e2v, dtype="int64")
        topo.create_dataset("surface_to_edge", data=s2e, dtype="int64")
        topo.create_dataset("cell_to_surface", data=c2s, dtype="int64")

        for name, count in [
            ("n_vertex", n_vertex),
            ("n_edge", n_edge),
            ("n_surface", n_surface),
            ("n_cell", n_cell),
        ]:
            topo.attrs[name] = np.int64(count)


def _build_csr(pairs, n_rows):
    """Build CSR from (row, col) pairs. col values are 1-based."""
    rows = defaultdict(list)
    for r, c in pairs:
        rows[r].append(c)
    indptr = [0]
    indices = []
    for r in range(n_rows):
        cols = sorted(set(rows[r]))
        indices.extend(cols)
        indptr.append(len(indices))
    return {
        "indptr": np.array(indptr, dtype=np.int64),
        "indices": np.array(indices, dtype=np.int64),
    }


def write_auxiliary(path, topology):
    """Write mesh_auxiliary.h5 with CSR adjacency relations."""
    e2v = topology["edge_to_vertex"]
    s2e = topology["surface_to_edge"]
    c2s = topology["cell_to_surface"]

    n_vertex = topology["vertex_to_coord"].shape[0]
    n_edge = e2v.shape[0]
    n_surface = s2e.shape[0]
    n_cell = c2s.shape[0]

    # surface_to_cell
    surface_to_cell = np.zeros((n_surface, 2), dtype=np.int64)
    for icell in range(n_cell):
        for s in c2s[icell]:
            sid = int(abs(s))
            if surface_to_cell[sid - 1, 0] == 0:
                surface_to_cell[sid - 1, 0] = icell + 1
            else:
                surface_to_cell[sid - 1, 1] = icell + 1

    # vertex_to_edge
    ve_pairs = []
    for eid in range(n_edge):
        v1, v2 = int(e2v[eid, 0]), int(e2v[eid, 1])
        ve_pairs.append((v1 - 1, eid + 1))
        ve_pairs.append((v2 - 1, eid + 1))

    # vertex_to_surface
    vs_pairs = []
    for sid in range(n_surface):
        for e in s2e[sid]:
            eid = int(abs(e))
            v1, v2 = int(e2v[eid - 1, 0]), int(e2v[eid - 1, 1])
            vs_pairs.append((v1 - 1, sid + 1))
            vs_pairs.append((v2 - 1, sid + 1))

    # vertex_to_cell
    vc_set = set()
    for icell, surfaces in enumerate(c2s):
        for s in surfaces:
            sid = int(abs(s))
            for e in s2e[sid - 1]:
                eid_abs = int(abs(e))
                v1, v2 = int(e2v[eid_abs - 1, 0]), int(e2v[eid_abs - 1, 1])
                vc_set.add((v1 - 1, icell + 1))
                vc_set.add((v2 - 1, icell + 1))
    vc_pairs = list(vc_set)

    # edge_to_surface
    es_pairs = []
    for sid in range(n_surface):
        for e in s2e[sid]:
            es_pairs.append((int(abs(e)) - 1, sid + 1))

    # edge_to_cell
    ec_set = set()
    for icell, surfaces in enumerate(c2s):
        for s in surfaces:
            sid = int(abs(s))
            for e in s2e[sid - 1]:
                ec_set.add((int(abs(e)) - 1, icell + 1))
    ec_pairs = list(ec_set)

    with h5py.File(path, "w") as f:
        aux = f.create_group("auxiliary")
        aux.create_dataset("surface_to_cell", data=surface_to_cell, dtype="int64")

        for name, pairs, n_rows in [
            ("vertex_to_edge", ve_pairs, n_vertex),
            ("vertex_to_surface", vs_pairs, n_vertex),
            ("vertex_to_cell", vc_pairs, n_vertex),
            ("edge_to_surface", es_pairs, n_edge),
            ("edge_to_cell", ec_pairs, n_edge),
        ]:
            csr = _build_csr(pairs, n_rows)
            g = aux.create_group(name)
            g.create_dataset("indptr", data=csr["indptr"], dtype="int64")
            g.create_dataset("indices", data=csr["indices"], dtype="int64")


def main():
    parser = argparse.ArgumentParser(description="GMSH to HDF5 topology converter")
    parser.add_argument("input", help="Input GMSH .msh file (v4.1)")
    parser.add_argument("-o", "--output", default="mesh.h5",
                        help="Output mesh.h5 (default: mesh.h5)")
    parser.add_argument("--aux", help="Optional auxiliary CSR file")
    args = parser.parse_args()

    mesh = meshio.read(args.input)
    topology = extract_topology(mesh)
    write_topology(args.output, topology)
    if args.aux:
        write_auxiliary(args.aux, topology)


if __name__ == "__main__":
    main()