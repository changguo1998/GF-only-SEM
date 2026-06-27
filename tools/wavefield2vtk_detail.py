#!/usr/bin/env python3
"""Convert all strain snapshot record files to VTK (mesh + GLL points).

Merges all MPI ranks, 3 source directions (x, y, z), and all time steps
into per-timestep VTK files. Includes hex cells (mesh corners) and GLL
vertex cells with per-point strain.

Usage:
    cd examples/halfspace/
    python ../../tools/wavefield2vtk_detail.py

Reads:
    mesh.h5
    wavefields/{x,y,z}/record_{r}.h5  (all ranks r)
    config.h5                       — for snapshot stride

Writes:
    vtk/wavefield_N.vtk              N = solver step number
"""

import glob
import os
import re

import h5py
import numpy as np

# ── Hex face definitions (local vertex indices, CCW from outside) ─────
_HEX_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]]

_VOIGT_LABELS = ["xx", "yy", "zz", "xy", "xz", "yz"]
_DIRECTIONS = ["x", "y", "z"]


def resolve_cell_vertices(cell_to_surface, surface_to_edge, edge_to_vertex, cell_idx):
    signed_surfaces = cell_to_surface[cell_idx]
    local_to_global = {}
    for fi in range(6):
        sid_signed = signed_surfaces[fi]
        sid = int(abs(sid_signed)) - 1
        canonical_edges = surface_to_edge[sid]
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
            eid = int(abs(signed_edges[k])) - 1
            gv1, gv2 = edge_to_vertex[eid]
            gv1 -= 1
            gv2 -= 1
            lvk = _HEX_FACES[fi][k]
            lvk_next = _HEX_FACES[fi][(k + 1) % 4]
            if signed_edges[k] > 0:
                local_to_global[lvk] = gv1
                local_to_global[lvk_next] = gv2
            else:
                local_to_global[lvk] = gv2
                local_to_global[lvk_next] = gv1
    return [local_to_global[lv] for lv in range(8)]


def build_global_connectivity(cell_to_surface, surface_to_edge, edge_to_vertex):
    n_cell = cell_to_surface.shape[0]
    connectivity = np.zeros((n_cell, 8), dtype=np.int64)
    for ci in range(n_cell):
        conn = resolve_cell_vertices(cell_to_surface, surface_to_edge, edge_to_vertex, ci)
        connectivity[ci] = conn
    return connectivity


def write_vtu(path, vertex_coords, connectivity, cell_fields, point_fields, n_mesh_vert):
    """Write VTK with hex cells + GLL vertex cells (binary).

    Points = [mesh_vertices | GLL_points].
    Hex cells reference mesh vertices (0..n_mesh_vert-1);
    GLL points (n_mesh_vert..) each get a VERTEX cell (type 1).
    """
    n_vert = vertex_coords.shape[0]
    n_hex = connectivity.shape[0]
    n_gll_pt = n_vert - n_mesh_vert
    total_cells = n_hex + n_gll_pt
    total_ints = n_hex * 9 + n_gll_pt * 2

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"wavefield snapshot converted to VTK (detail)\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET UNSTRUCTURED_GRID\n")

        # ── Points ──
        f.write(f"POINTS {n_vert} float\n".encode())
        f.write(np.ascontiguousarray(vertex_coords, dtype=">f4").tobytes())
        f.write(b"\n")

        # ── Cells: hex + GLL vertex ──
        f.write(f"CELLS {total_cells} {total_ints}\n".encode())
        hex_arr = np.zeros(n_hex * 9, dtype=np.int32)
        hex_arr[0::9] = 8
        for i in range(8):
            hex_arr[1 + i :: 9] = connectivity[:, i].astype(np.int32)
        gll_arr = np.zeros(n_gll_pt * 2, dtype=np.int32)
        gll_arr[0::2] = 1
        gll_arr[1::2] = np.arange(n_mesh_vert, n_vert, dtype=np.int32)
        cell_arr = np.concatenate([hex_arr, gll_arr])
        f.write(np.ascontiguousarray(cell_arr, dtype=">i4").tobytes())
        f.write(b"\n")

        # ── Cell types ──
        f.write(f"CELL_TYPES {total_cells}\n".encode())
        types_arr = np.concatenate(
            [np.full(n_hex, 12, dtype=np.int32), np.full(n_gll_pt, 1, dtype=np.int32)]
        )
        f.write(np.ascontiguousarray(types_arr, dtype=">i4").tobytes())
        f.write(b"\n")

        # ── Cell data ──
        f.write(f"CELL_DATA {n_hex}\n".encode())
        for name, data in cell_fields.items():
            f.write(f"SCALARS {name} float 1\n".encode())
            f.write(b"LOOKUP_TABLE default\n")
            f.write(np.ascontiguousarray(data, dtype=">f4").tobytes())
            f.write(b"\n")

        # ── Point data ──
        f.write(f"POINT_DATA {n_vert}\n".encode())
        for name, data in point_fields.items():
            f.write(f"SCALARS {name} float 1\n".encode())
            f.write(b"LOOKUP_TABLE default\n")
            f.write(np.ascontiguousarray(data, dtype=">f4").tobytes())
            f.write(b"\n")


def find_record_files(wave_dir):
    pattern = os.path.join(wave_dir, "record_*.h5")
    files = glob.glob(pattern)
    files.sort(key=lambda p: int(re.search(r"record_(\d+)\.h5$", p).group(1)))
    return files


def main():
    cwd = os.getcwd()
    mesh_path = os.path.join(cwd, "mesh.h5")
    config_path = os.path.join(cwd, "config.h5")

    # ── Read mesh topology + GLL coords ──
    print(f"[wavefield2vtk_detail] Reading {mesh_path}")
    with h5py.File(mesh_path, "r") as f:
        topo = f["topology"]
        vertex_to_coord = topo["vertex_to_coord"][:]
        edge_to_vertex = topo["edge_to_vertex"][:]
        surface_to_edge = topo["surface_to_edge"][:]
        cell_to_surface = topo["cell_to_surface"][:]
        is_pml = np.zeros(cell_to_surface.shape[0], dtype=np.int8)
        if "field/element/is_pml" in f:
            is_pml[:] = f["field/element/is_pml"][:]
        gll_coords = f["field/element/coords"][:]  # (n_cell, ngll, ngll, ngll, 3)
    n_cell = cell_to_surface.shape[0]
    n_mesh_vert = vertex_to_coord.shape[0]
    ngll = gll_coords.shape[1]
    print(f"  Global cells: {n_cell}, vertices: {n_mesh_vert}, NGLL: {ngll}")

    # ── Build global hex connectivity ──
    print("[wavefield2vtk_detail] Resolving hexahedral connectivity...")
    connectivity = build_global_connectivity(cell_to_surface, surface_to_edge, edge_to_vertex)

    # ── Assemble GLL point array (same for all timesteps) ──
    # Points = [all_mesh_vertices | GLL_points_of_cell_0 | GLL_points_of_cell_1 | ...]
    # Build once, reuse for every timestep
    print("[wavefield2vtk_detail] Building GLL point array...")
    gll_pt_list = []
    for ci in range(n_cell):
        gll_pt_list.append(gll_coords[ci].reshape(-1, 3))
    gll_points = np.concatenate(gll_pt_list, axis=0) if gll_pt_list else np.empty((0, 3))
    all_points = np.concatenate([vertex_to_coord, gll_points], axis=0)
    gll_per_cell = ngll**3
    total_gll = n_cell * gll_per_cell
    print(f"  GLL points per cell: {gll_per_cell}, total GLL: {total_gll}")
    print(f"  Total points: {all_points.shape[0]}")

    # ── Find record files per direction ──
    record_paths = {}
    for d in _DIRECTIONS:
        wave_dir = os.path.join(cwd, f"wavefields/{d}")
        files = find_record_files(wave_dir)
        if not files:
            print(f"[wavefield2vtk_detail] Error: no record_*.h5 files in {wave_dir}")
            return 1
        record_paths[d] = files
        print(f"  wavefields/{d}/: {len(files)} rank files")

    # ── Read metadata ──
    with h5py.File(record_paths["x"][0], "r") as f:
        n_snapshots = f["strain"].shape[0]
    print(f"  Snapshots: {n_snapshots}")

    stride = 1
    if os.path.isfile(config_path):
        try:
            with h5py.File(config_path, "r") as f:
                stride = int(f["simulation"].attrs.get("snapshot_stride", 1))
        except Exception:
            pass
    print(f"  Snapshot stride: {stride}")

    # ── Pre-read local_element_ids ──
    local_eids_list = []
    for path in record_paths["x"]:
        with h5py.File(path, "r") as f:
            local_eids_list.append(f["local_element_ids"][:].copy())
    for d in ("y", "z"):
        for ri, path in enumerate(record_paths[d]):
            with h5py.File(path, "r") as f:
                eids = f["local_element_ids"][:]
                if not np.array_equal(eids, local_eids_list[ri]):
                    print(f"[wavefield2vtk_detail] Error: element ID mismatch in {path}")
                    return 1

    # ── Open all record files ──
    files = {}
    for d in _DIRECTIONS:
        files[d] = [h5py.File(p, "r") for p in record_paths[d]]

    strain_field_names = [f"strain_{vl}_{d}" for d in _DIRECTIONS for vl in _VOIGT_LABELS]
    cell_fields_pml = {"PML_flag": is_pml.astype(np.float64)}

    out_dir = os.path.join(cwd, "vtk")
    os.makedirs(out_dir, exist_ok=True)

    # ── Iterate snapshots ──
    for snap_idx in range(n_snapshots):
        step_num = snap_idx * stride

        # Build per-element GLL strain for all 3 directions
        # shape per direction: (n_cell, ngll^3, 6)
        dir_strain = {}
        for d in _DIRECTIONS:
            gs = np.zeros((n_cell, gll_per_cell, 6), dtype=np.float64)
            for ri, f in enumerate(files[d]):
                snap = f["strain"][snap_idx]  # (n_local, ngll, ngll, ngll, 6)
                gs[local_eids_list[ri]] = snap.reshape(-1, gll_per_cell, 6)
            dir_strain[d] = gs

        # Build point data: 18 strain fields
        # mesh vertex strain = 0; GLL points get their per-point values
        point_fields = {}
        for fi, name in enumerate(strain_field_names):
            di = fi // 6
            ci = fi % 6
            arr = np.zeros(all_points.shape[0], dtype=np.float64)
            # Flatten per-element GLL strain into contiguous GLL block
            # GLL block layout: [cell0_gll0..cell0_gllM, cell1_gll0.., ...]
            # dir_strain[di][:, :, ci] shape (n_cell, gll_per_cell)
            arr[n_mesh_vert:] = dir_strain[_DIRECTIONS[di]][:, :, ci].ravel()
            point_fields[name] = arr

        out_path = os.path.join(out_dir, f"wavefield_{step_num}.vtk")
        print(f"[wavefield2vtk_detail] Writing {out_path}")
        write_vtu(out_path, all_points, connectivity, cell_fields_pml, point_fields, n_mesh_vert)

    # ── Cleanup ──
    for d in _DIRECTIONS:
        for f in files[d]:
            f.close()

    print(f"  Done. {n_snapshots} files written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
