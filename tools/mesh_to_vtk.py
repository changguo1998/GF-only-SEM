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
            local_ids = f["partition/local_element_ids"][:]  # global 1-based
            for name in field_names:
                data = f[f"field/element/{name}"][:]  # (n_local, NGLL, NGLL, NGLL)
                avg = np.mean(data, axis=(1, 2, 3))  # per-element average
                for li, gid in enumerate(local_ids):
                    fields[name][gid - 1] = avg[li]

    return fields


# ── VTK writer (legacy ASCII) ──────────────────────────────────────────


def write_vtu(path, vertex_coords, connectivity, cell_fields):
    """Write VTK Unstructured Grid (legacy ASCII .vtk format).

    Args:
        vertex_coords: (n_vertex, 3) float64
        connectivity:  (n_cell, 8) int64  — VTK hex vertex indices (0-based)
        cell_fields:   dict of name→(n_cell,) float64  — cell data arrays
    """
    n_vert = vertex_coords.shape[0]
    n_cell = connectivity.shape[0]

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mesh.h5 converted to VTK\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # ── Points ──
        f.write(f"POINTS {n_vert} float\n")
        for v in vertex_coords:
            f.write(f"  {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")

        # ── Cells ──
        # VTK hex: type 12, 8 vertices per cell
        f.write(f"CELLS {n_cell} {n_cell * 9}\n")
        for conn in connectivity:
            f.write("  8 " + " ".join(str(int(c)) for c in conn) + "\n")

        f.write(f"CELL_TYPES {n_cell}\n")
        for _ in range(n_cell):
            f.write("  12\n")

        # ── Cell data ──
        f.write(f"CELL_DATA {n_cell}\n")
        for name, data in cell_fields.items():
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in data:
                f.write(f"  {val:.8e}\n")


# ── Main ───────────────────────────────────────────────────────────────


def main():
    cwd = os.getcwd()
    mesh_path = os.path.join(cwd, "mesh.h5")
    out_path = os.path.join(cwd, "mesh.vtk")
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

    # ── Write VTK ──
    print(f"[mesh_to_vtk] Writing {out_path}")
    write_vtu(out_path, vertex_to_coord, connectivity, cell_fields)
    print("  Done.")


if __name__ == "__main__":
    main()
