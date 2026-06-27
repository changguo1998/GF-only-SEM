#!/usr/bin/env python3
"""Convert mesh.h5 (+ partitions/) in CWD to mesh.vtk.

Reads mesh.h5 topology and partition material fields from the current
working directory, resolves hexahedral cell connectivity, and writes
mesh.vtk with cell-averaged Vp, Vs, density, mass, PML damping.
Viewable in ParaView / VisIt.
"""

import os

import h5py
import numpy as np

# ── Hex face definitions (local vertex indices, CCW from outside) ─────
# GMSH hex ordering (also VTK hex ordering):
#   v0(0,0,0) v1(1,0,0) v2(1,1,0) v3(0,1,0)  — bottom
#   v4(0,0,1) v5(1,0,1) v6(1,1,1) v7(0,1,1)  — top
_HEX_FACES = [
    [0, 3, 2, 1],  # -z (bottom)
    [4, 5, 6, 7],  # +z (top)
    [0, 1, 5, 4],  # -y (front)
    [3, 7, 6, 2],  # +y (back)
    [0, 4, 7, 3],  # -x (left)
    [1, 2, 6, 5],  # +x (right)
]


# ── Topology resolution ────────────────────────────────────────────────


def resolve_cell_vertices(cell_to_surface, surface_to_edge, edge_to_vertex, cell_idx):
    """Resolve the 8 global vertex indices of a hex cell.

    Uses the signed surface/edge topology to build a mapping from
    local hex vertex index (0-7) to global vertex index (0-based).

    Returns list of 8 global vertex indices in VTK hex order:
        [v0, v1, v2, v3, v4, v5, v6, v7]
    """
    signed_surfaces = cell_to_surface[cell_idx]  # (6,) 1-based, signed
    local_to_global = {}  # lv → gv

    for fi in range(6):
        sid_signed = signed_surfaces[fi]  # signed surface id (+ or -)
        sid = int(abs(sid_signed)) - 1  # 0-based surface id
        canonical_edges = surface_to_edge[sid]  # canonical (4,) 1-based, signed

        # If cell uses reversed orientation, negate and reverse edge order
        if sid_signed > 0:
            signed_edges = canonical_edges
        else:
            signed_edges = [
                -canonical_edges[3],
                -canonical_edges[2],
                -canonical_edges[1],
                -canonical_edges[0],
            ]

        for k in range(4):
            eid = int(abs(signed_edges[k])) - 1  # 0-based edge id
            gv1, gv2 = edge_to_vertex[eid]  # 1-based, sorted: gv1 < gv2
            gv1 -= 1
            gv2 -= 1

            lvk = _HEX_FACES[fi][k]
            lvk_next = _HEX_FACES[fi][(k + 1) % 4]

            if signed_edges[k] > 0:
                # Edge direction matches CCW face traversal:
                #   gv1 → lvk,  gv2 → lvk_next
                local_to_global[lvk] = gv1
                local_to_global[lvk_next] = gv2
            else:
                # Edge direction reversed:
                #   gv2 → lvk,  gv1 → lvk_next
                local_to_global[lvk] = gv2
                local_to_global[lvk_next] = gv1

    return [local_to_global[lv] for lv in range(8)]


def build_cell_connectivity(cell_to_surface, surface_to_edge, edge_to_vertex):
    """Build full [n_cell, 8] connectivity array (0-based vertex indices)."""
    n_cell = cell_to_surface.shape[0]
    connectivity = np.zeros((n_cell, 8), dtype=np.int64)
    for ci in range(n_cell):
        conn = resolve_cell_vertices(cell_to_surface, surface_to_edge, edge_to_vertex, ci)
        connectivity[ci] = conn
    return connectivity


# ── Field assembly from partition files ────────────────────────────────


def read_partition_fields(partition_dir, n_cell):
    """Aggregate element fields from all partition_{r}.h5 files.

    Returns dict with keys:
      vp, vs, density, mass, damping
    Each value has shape (n_cell,) containing the element-averaged value.
    is_pml has shape (n_cell,) int8 (from cell-level field, not averaged).
    """
    # Find all partition files
    part_files = sorted(
        f for f in os.listdir(partition_dir) if f.startswith("partition_") and f.endswith(".h5")
    )
    if not part_files:
        raise FileNotFoundError(f"No partition_*.h5 files found in {partition_dir}")

    # Read element_to_rank from first partition file
    with h5py.File(os.path.join(partition_dir, part_files[0]), "r") as f:
        elem_to_rank = f["partition/element_to_rank"][:]  # (n_cell,)

    # Aggregate fields: vp, vs, density, mass, damping
    # Each stored per-element with shape (n_local, NGLL, NGLL, NGLL)
    field_names = ["vp", "vs", "density", "mass", "damping"]
    fields = {name: np.zeros(n_cell, dtype=np.float64) for name in field_names}

    for pf in part_files:
        with h5py.File(os.path.join(partition_dir, pf), "r") as f:
            local_ids = f["partition/local_element_ids"][:]  # 0-based
            for name in field_names:
                data = f[f"field/element/{name}"][:]  # (n_local, NGLL, NGLL, NGLL)
                avg = np.mean(data, axis=(1, 2, 3))  # per-element average
                for li, gid in enumerate(local_ids):
                    fields[name][gid] = avg[li]

    return fields


# ── VTK writer (legacy binary) ──────────────────────────────────────────


def write_vtu(path, vertex_coords, connectivity, cell_fields, point_fields=None, n_mesh_vert=None):
    """Write VTK Unstructured Grid (legacy binary .vtk format).

    If *point_fields* is provided, writes GLL vertex cells (detail mode).
    Otherwise writes mesh-only (hex cells only).

    Args:
        vertex_coords: (n_vertex, 3) float64
        connectivity:  (n_cell, 8) int64  — VTK hex vertex indices (0-based)
        cell_fields:   dict of name→(n_cell,) float64  — cell data arrays
        point_fields:  optional dict of name→(n_vertex,) float64  — point data arrays
        n_mesh_vert:   int, number of mesh vertices (required if point_fields given)
    """
    n_vert = vertex_coords.shape[0]
    n_hex = connectivity.shape[0]
    has_detail = point_fields is not None
    if has_detail:
        n_gll_pt = n_vert - n_mesh_vert
        total_cells = n_hex + n_gll_pt
        total_ints = n_hex * 9 + n_gll_pt * 2
    else:
        total_cells = n_hex
        total_ints = n_hex * 9

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"mesh.h5 converted to VTK\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET UNSTRUCTURED_GRID\n")

        # ── Points (float32 big-endian) ──
        f.write(f"POINTS {n_vert} float\n".encode())
        f.write(np.ascontiguousarray(vertex_coords, dtype=">f4").tobytes())
        f.write(b"\n")

        # ── Cells: hex + optional GLL vertex cells ──
        f.write(f"CELLS {total_cells} {total_ints}\n".encode())
        hex_arr = np.zeros(n_hex * 9, dtype=np.int32)
        hex_arr[0::9] = 8  # size prefix
        for i in range(8):
            hex_arr[1 + i :: 9] = connectivity[:, i].astype(np.int32)
        if has_detail:
            gll_arr = np.zeros(n_gll_pt * 2, dtype=np.int32)
            gll_arr[0::2] = 1
            gll_arr[1::2] = np.arange(n_mesh_vert, n_vert, dtype=np.int32)
            cell_arr = np.concatenate([hex_arr, gll_arr])
        else:
            cell_arr = hex_arr
        f.write(np.ascontiguousarray(cell_arr, dtype=">i4").tobytes())
        f.write(b"\n")

        # ── Cell types (12 = VTK_HEXAHEDRON, 1 = VTK_VERTEX) ──
        f.write(f"CELL_TYPES {total_cells}\n".encode())
        if has_detail:
            types_arr = np.concatenate(
                [np.full(n_hex, 12, dtype=np.int32), np.full(n_gll_pt, 1, dtype=np.int32)]
            )
        else:
            types_arr = np.full(n_hex, 12, dtype=np.int32)
        f.write(np.ascontiguousarray(types_arr, dtype=">i4").tobytes())
        f.write(b"\n")

        # ── Cell data (hex cells only) ──
        f.write(f"CELL_DATA {n_hex}\n".encode())
        for name, data in cell_fields.items():
            f.write(f"SCALARS {name} float 1\n".encode())
            f.write(b"LOOKUP_TABLE default\n")
            f.write(np.ascontiguousarray(data, dtype=">f4").tobytes())
            f.write(b"\n")

        # ── Point data (detail mode only) ──
        if has_detail:
            f.write(f"POINT_DATA {n_vert}\n".encode())
            for name, data in point_fields.items():
                f.write(f"SCALARS {name} float 1\n".encode())
                f.write(b"LOOKUP_TABLE default\n")
                f.write(np.ascontiguousarray(data, dtype=">f4").tobytes())
                f.write(b"\n")


# ── Main ───────────────────────────────────────────────────────────────


def main():
    cwd = os.getcwd()
    mesh_path = os.path.join(cwd, "mesh.h5")
    vtk_dir = os.path.join(cwd, "vtk")
    out_path = os.path.join(vtk_dir, "mesh.vtk")
    part_dir = os.path.join(cwd, "partitions")

    has_partitions = os.path.isdir(part_dir)

    # ── Read topology ──
    print(f"[mesh_to_vtk] Reading {mesh_path}")
    with h5py.File(mesh_path, "r") as f:
        topo = f["topology"]
        vertex_to_coord = topo["vertex_to_coord"][:]
        edge_to_vertex = topo["edge_to_vertex"][:]
        surface_to_edge = topo["surface_to_edge"][:]
        cell_to_surface = topo["cell_to_surface"][:]

        is_pml = np.zeros(cell_to_surface.shape[0], dtype=np.int8)
        if "field/element/is_pml" in f:
            is_pml[:] = f["field/element/is_pml"][:]

        gll_coords = f["field/element/coords"][:] if "field/element/coords" in f else None

    n_cell = cell_to_surface.shape[0]
    n_vert = vertex_to_coord.shape[0]
    print(f"  Cells: {n_cell}, Vertices: {n_vert}")

    # ── Resolve connectivity ──
    print("[mesh_to_vtk] Resolving hexahedral connectivity...")
    connectivity = build_cell_connectivity(cell_to_surface, surface_to_edge, edge_to_vertex)

    # ── Read partition fields ──
    cell_fields = {}
    if has_partitions:
        print("[mesh_to_vtk] Reading partitions/...")
        fields = read_partition_fields(part_dir, n_cell)
        cell_fields["Vp_m_s"] = fields["vp"]
        cell_fields["Vs_m_s"] = fields["vs"]
        cell_fields["Density_kg_m3"] = fields["density"]
        cell_fields["Mass"] = fields["mass"]
        cell_fields["PML_Damping"] = fields["damping"]
        cell_fields["PML_flag"] = is_pml.astype(np.float64)
        print("  Fields: " + ", ".join(cell_fields.keys()))
    else:
        cell_fields["PML_flag"] = is_pml.astype(np.float64)
        print("  Fields: PML_flag only (no partitions/)")

    # ── Build GLL point data if available ──
    point_fields = None
    n_mesh_vert = None
    vertex_coords_out = vertex_to_coord
    if gll_coords is not None:
        print("[mesh_to_vtk] Building GLL point data...")
        ngll = gll_coords.shape[1]
        gll_per_cell = ngll**3
        n_mesh_vert = n_vert

        # Combined point array: mesh vertices + GLL points
        gll_pt_list = []
        for ci in range(n_cell):
            gll_pt_list.append(gll_coords[ci].reshape(-1, 3))
        gll_points = np.concatenate(gll_pt_list, axis=0) if gll_pt_list else np.empty((0, 3))
        vertex_coords_out = np.concatenate([vertex_to_coord, gll_points], axis=0)

        # Point fields: mesh vertices → 0, GLL points → cell's averaged value
        point_fields = {}
        for name, data in cell_fields.items():
            arr = np.zeros(vertex_coords_out.shape[0], dtype=np.float64)
            for ci in range(n_cell):
                s = n_mesh_vert + ci * gll_per_cell
                arr[s : s + gll_per_cell] = data[ci]
            point_fields[name] = arr

        print(f"  GLL per cell: {gll_per_cell}, total GLL: {n_cell * gll_per_cell}")
        print(f"  Total points: {vertex_coords_out.shape[0]}")

    # ── Write VTK ──
    os.makedirs(vtk_dir, exist_ok=True)
    print(f"[mesh_to_vtk] Writing {out_path}")
    write_vtu(out_path, vertex_coords_out, connectivity, cell_fields, point_fields, n_mesh_vert)
    print("  Done.")


if __name__ == "__main__":
    main()
