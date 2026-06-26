#!/usr/bin/env python3
"""Convert partitioned mesh (partitions/partition_{r}.h5) to per-rank VTK files.

Reads mesh.h5 topology and each partition file from the partitions/ directory
in CWD, resolves hexahedral cell connectivity for local elements, and writes
partition_{r}.vtk per rank with cell-averaged Vp, Vs, density, mass, damping,
PML flag, and rank assignment.

Useful for visually inspecting METIS partition quality and per-rank element
distribution in ParaView / VisIt.
"""

import os
import re

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


def build_local_connectivity(local_element_ids, cell_to_surface, surface_to_edge, edge_to_vertex):
    """Build [n_local, 8] connectivity for local elements (0-based vertex indices)."""
    n_local = len(local_element_ids)
    connectivity = np.zeros((n_local, 8), dtype=np.int64)
    for li, gid in enumerate(local_element_ids):
        conn = resolve_cell_vertices(cell_to_surface, surface_to_edge, edge_to_vertex, gid)
        connectivity[li] = conn
    return connectivity


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
        f.write("partition file converted to VTK\n")
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
    part_dir = os.path.join(cwd, "partitions")
    vtk_dir = os.path.join(cwd, "vtk")

    if not os.path.isdir(part_dir):
        print("[partition_to_vtk] Error: no partitions/ directory found in CWD")
        return 1

    # ── Read topology from mesh.h5 ──
    print(f"[partition_to_vtk] Reading {mesh_path}")
    with h5py.File(mesh_path, "r") as f:
        topo = f["topology"]
        vertex_to_coord = topo["vertex_to_coord"][:]
        edge_to_vertex = topo["edge_to_vertex"][:]
        surface_to_edge = topo["surface_to_edge"][:]
        cell_to_surface = topo["cell_to_surface"][:]

        is_pml_global = np.zeros(cell_to_surface.shape[0], dtype=np.int8)
        if "field/element/is_pml" in f:
            is_pml_global[:] = f["field/element/is_pml"][:]

    n_cell = cell_to_surface.shape[0]
    n_vert = vertex_to_coord.shape[0]
    print(f"  Global cells: {n_cell}, vertices: {n_vert}")

    # ── Find all partition files ──
    part_files = sorted(
        f for f in os.listdir(part_dir) if f.startswith("partition_") and f.endswith(".h5")
    )
    if not part_files:
        print("[partition_to_vtk] Error: no partition_*.h5 files in partitions/")
        return 1

    print(f"[partition_to_vtk] Found {len(part_files)} partition files")

    os.makedirs(vtk_dir, exist_ok=True)

    # ── Process each partition ──
    for pf in part_files:
        # Extract rank from filename
        m = re.match(r"partition_(\d+)\.h5$", pf)
        if not m:
            continue
        rank = int(m.group(1))
        part_path = os.path.join(part_dir, pf)
        out_path = os.path.join(vtk_dir, f"partition_{rank}.vtk")

        with h5py.File(part_path, "r") as f:
            local_zero = f["partition/local_element_ids"][:]  # 0-based
            n_local = len(local_zero)

            # Build connectivity for local elements
            connectivity = build_local_connectivity(
                local_zero, cell_to_surface, surface_to_edge, edge_to_vertex
            )

            # Read field data: cell-average GLL-node fields
            field_names = ["vp", "vs", "density", "mass", "damping"]
            cell_fields = {}
            for name in field_names:
                data = f[f"field/element/{name}"][:]  # (n_local, NGLL, NGLL, NGLL)
                cell_fields[name] = np.mean(data, axis=(1, 2, 3))

            # PML flag from global (subset to local elements)
            cell_fields["PML_flag"] = is_pml_global[local_zero].astype(np.float64)

            # Rank field (for coloring by partition)
            cell_fields["Rank"] = np.full(n_local, float(rank))

        # ── Write VTK ──
        # Rename for display
        vtk_fields = {
            "Vp_m_s": cell_fields["vp"],
            "Vs_m_s": cell_fields["vs"],
            "Density_kg_m3": cell_fields["density"],
            "Mass": cell_fields["mass"],
            "PML_Damping": cell_fields["damping"],
            "PML_flag": cell_fields["PML_flag"],
            "Rank": cell_fields["Rank"],
        }

        print(f"  [{pf}] {n_local} cells → {out_path}")
        write_vtu(out_path, vertex_to_coord, connectivity, vtk_fields)

    print("[partition_to_vtk] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
