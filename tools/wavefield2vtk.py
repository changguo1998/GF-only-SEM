#!/usr/bin/env python3
"""Convert strain snapshot record files to VTK (cell-corner strain).

Merges all MPI ranks, 3 source directions (x, y, z), and all time steps
into per-timestep VTK files. Strain is recorded at mesh vertices; this
tool maps values back to element corners via mesh topology and averages
the 8 corners to produce cell-centered data.

Usage:
    cd examples/halfspace/
    wavefield2vtk

Reads:
    model.h5
    config.h5                          — snapshot stride, nsteps
    wavefields/{x,y,z}/record_{r}.h5   — strain at recorded vertices

Writes:
    vtk/wavefield_N.vtk                — per-timestep VTK (cell data)
"""

import glob
import argparse
import os
import re

import h5py
import numpy as np

_VOIGT_LABELS = ["xx", "yy", "zz", "xy", "xz", "yz"]
_DIRECTIONS = ["x", "y", "z"]


def build_element_vertex_map(cell_to_surface, surface_to_edge, edge_to_vertex):
    """Build [n_cell, 8] array of 0-based global vertex IDs for each hex element.

    Uses GMSH-like topology: each hex has 6 signed surfaces, each surface
    has 4 signed edges, each edge has 2 vertices. Returns 0-based indices
    suitable for directly indexing vertex_to_coord.
    """
    _HEX_FACES = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 7, 6, 2],
        [0, 4, 7, 3],
        [1, 2, 6, 5],
    ]

    n_cell = cell_to_surface.shape[0]
    connectivity = np.zeros((n_cell, 8), dtype=np.int64)

    for ci in range(n_cell):
        signed_surfaces = cell_to_surface[ci]
        local_to_global = {}
        for fi in range(6):
            sid_signed = signed_surfaces[fi]
            sid = int(abs(sid_signed)) - 1
            canonical_edges = surface_to_edge[sid]
            if sid_signed > 0:
                signed_edges = canonical_edges
            else:
                # Reverse face orientation
                signed_edges = [
                    -canonical_edges[3],
                    -canonical_edges[2],
                    -canonical_edges[1],
                    -canonical_edges[0],
                ]
            for k in range(4):
                eid = int(abs(signed_edges[k])) - 1
                gv1, gv2 = edge_to_vertex[eid]
                gv1 -= 1  # to 0-based
                gv2 -= 1
                lvk = _HEX_FACES[fi][k]
                lvk_next = _HEX_FACES[fi][(k + 1) % 4]
                if signed_edges[k] > 0:
                    local_to_global[lvk] = gv1
                    local_to_global[lvk_next] = gv2
                else:
                    local_to_global[lvk] = gv2
                    local_to_global[lvk_next] = gv1
        connectivity[ci] = [local_to_global[lv] for lv in range(8)]
    return connectivity


def find_record_files(wave_dir):
    """Return sorted list of record file paths for a wavefield direction."""
    pattern = os.path.join(wave_dir, "record_*.h5")
    files = glob.glob(pattern)
    files.sort(key=lambda p: int(re.search(r"record_(\d+)\.h5$", p).group(1)))
    return files


def write_vtu(path, vertex_coords, connectivity, cell_fields):
    """Write legacy VTK (v3.0 unstructured grid) with cell data."""
    n_vert = vertex_coords.shape[0]
    n_cell = connectivity.shape[0]

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"wavefield snapshot converted to VTK\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET UNSTRUCTURED_GRID\n")

        # ── Points ──
        f.write(f"POINTS {n_vert} float\n".encode())
        f.write(np.ascontiguousarray(vertex_coords, dtype=">f4").tobytes())
        f.write(b"\n")

        # ── Cells ──
        f.write(f"CELLS {n_cell} {n_cell * 9}\n".encode())
        cell_arr = np.zeros(n_cell * 9, dtype=np.int32)
        cell_arr[0::9] = 8
        for i in range(8):
            cell_arr[1 + i :: 9] = connectivity[:, i].astype(np.int32)
        f.write(np.ascontiguousarray(cell_arr, dtype=">i4").tobytes())
        f.write(b"\n")

        # ── Cell types ──
        f.write(f"CELL_TYPES {n_cell}\n".encode())
        f.write(np.full(n_cell, 12, dtype=">i4").tobytes())
        f.write(b"\n")

        # ── Cell data ──
        f.write(f"CELL_DATA {n_cell}\n".encode())
        for name, data in cell_fields.items():
            f.write(f"SCALARS {name} float 1\n".encode())
            f.write(b"LOOKUP_TABLE default\n")
            f.write(np.ascontiguousarray(data, dtype=">f4").tobytes())
            f.write(b"\n")


def main(verbose: bool = False):
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "model.h5")
    config_path = os.path.join(cwd, "config.h5")
    parts_dir = os.path.join(cwd, "partitions")
    # ── Read mesh topology ──
    print(f"[wavefield2vtk] Reading {model_path}")
    with h5py.File(model_path, "r") as f:
        topo = f["topology"]
        vertex_to_coord = topo["vertex_to_coord"][:]
        edge_to_vertex = topo["edge_to_vertex"][:]
        surface_to_edge = topo["surface_to_edge"][:]
        cell_to_surface = topo["cell_to_surface"][:]
        is_pml = np.zeros(cell_to_surface.shape[0], dtype=np.int8)
        if "field/element/is_pml" in f:
            is_pml[:] = f["field/element/is_pml"][:]
    n_cell = cell_to_surface.shape[0]
    print(f"  Global cells: {n_cell}, vertices: {vertex_to_coord.shape[0]}")

    if verbose:
        print("[wavefield2vtk] Resolving hexahedral connectivity...")
    connectivity = build_element_vertex_map(cell_to_surface, surface_to_edge, edge_to_vertex)

    # ── Find record files per direction ──
    record_paths = {}
    for d in _DIRECTIONS:
        wave_dir = os.path.join(cwd, f"wavefields/{d}")
        files = find_record_files(wave_dir)
        if not files:
            print(f"[wavefield2vtk] Error: no record_*.h5 files in {wave_dir}")
            return 1
        record_paths[d] = files
        if verbose:
            print(f"  wavefields/{d}/: {len(files)} rank files")

    # ── Read metadata — find first rank with recording data ──
    n_snapshots = 0
    for rec_path in record_paths["x"]:
        with h5py.File(rec_path, "r") as f:
            ns = f["strain"].shape[0]
            if ns > 0:
                n_snapshots = ns
                break
    if n_snapshots == 0:
        print("[wavefield2vtk] Error: no snapshots in any record file")
        return 1
    print(f"  Snapshots: {n_snapshots}")

    # ── Read snapshot stride from config.h5 ──
    stride = 1
    if os.path.isfile(config_path):
        try:
            with h5py.File(config_path, "r") as f:
                stride = int(f["config"].attrs["snapshot_stride"])
        except Exception:
            pass
    if verbose:
        print(f"  Snapshot stride: {stride}")

    # ── Pre-read vertex IDs per rank file (must match across directions) ──
    vertex_id_list = []
    for path in record_paths["x"]:
        with h5py.File(path, "r") as f:
            vertex_id_list.append(f["vertex_ids"][:].copy())
    for d in ("y", "z"):
        for ri, path in enumerate(record_paths[d]):
            with h5py.File(path, "r") as f:
                vids = f["vertex_ids"][:]
                if not np.array_equal(vids, vertex_id_list[ri]):
                    print(f"[wavefield2vtk] Error: vertex ID mismatch in {path}")
                    return 1

    # ── Open all record files ──
    files = {}
    for d in _DIRECTIONS:
        files[d] = [h5py.File(p, "r") for p in record_paths[d]]

    # ── Build vertex_index → global_vertex_id mapping ──
    # vertex_id_list[ri] is the vertex_ids array for rank ri.
    # Different ranks may have overlapping vertex_ids (shared vertices).
    # We'll accumulate and average later.

    strain_field_names = [f"strain_{vl}_{d}" for d in _DIRECTIONS for vl in _VOIGT_LABELS]

    out_dir = os.path.join(cwd, "vtk")
    os.makedirs(out_dir, exist_ok=True)

    # ── Iterate snapshots ──
    for snap_idx in range(n_snapshots):
        step_num = snap_idx * stride

        # Build vertex → strain maps for all 3 directions
        # vertex_strain[d][vertex_id] = [strain_xx, ..., strain_yz]
        vertex_strain = [{} for _ in range(3)]

        for di, d in enumerate(_DIRECTIONS):
            for ri, f in enumerate(files[d]):
                vids = vertex_id_list[ri]  # (n_vertices,)
                if len(vids) == 0:
                    continue
                strain_snap = f["strain"][snap_idx]  # (n_vertices, 6)
                for vi in range(len(vids)):
                    vid = int(vids[vi])
                    if vid not in vertex_strain[di]:
                        vertex_strain[di][vid] = []
                    vertex_strain[di][vid].append(strain_snap[vi])

        # Average duplicate vertex entries (shared across ranks)
        for di in range(3):
            for vid in vertex_strain[di]:
                arr = np.array(vertex_strain[di][vid], dtype=np.float64)
                vertex_strain[di][vid] = arr.mean(axis=0)  # (6,)

        # For each element, look up strain at its 8 corner vertices.
        # Average available corners to produce cell strain.
        # Only elements with at least 1 recorded corner get non-zero data.
        dir_strain = np.zeros((3, n_cell, 6), dtype=np.float64)
        n_corners = np.zeros(n_cell, dtype=np.int32)

        for ci in range(n_cell):
            corner_vids = connectivity[ci]  # 8 vertex IDs (0-based)
            for corner_idx in range(8):
                gvid = int(corner_vids[corner_idx])
                for di in range(3):
                    if gvid in vertex_strain[di]:
                        dir_strain[di, ci] += vertex_strain[di][gvid]
                n_corners[ci] += 1 if any(gvid in vertex_strain[di] for di in range(3)) else 0

        # Average: dir_strain currently holds sum of corner values
        for ci in range(n_cell):
            if n_corners[ci] > 0:
                for di in range(3):
                    dir_strain[di, ci] /= n_corners[ci]

        # Build cell field dict
        cell_fields = {}
        for fi, name in enumerate(strain_field_names):
            di = fi // 6
            ci = fi % 6
            cell_fields[name] = dir_strain[di, :, ci]
        cell_fields["PML_flag"] = is_pml.astype(np.float64)
        cell_fields["n_recorded_corners"] = n_corners.astype(np.float64)

        out_path = os.path.join(out_dir, f"wavefield_{step_num}.vtk")

        if verbose:
            print(f"[wavefield2vtk] Writing {out_path}")
        write_vtu(out_path, vertex_to_coord, connectivity, cell_fields)

    # ── Cleanup ──
    for d in _DIRECTIONS:
        for f in files[d]:
            f.close()

    print(f"  Done. {n_snapshots} files written to {out_dir}/")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert strain snapshots to per-timestep VTK files.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed processing messages")
    args = parser.parse_args()
    raise SystemExit(main(verbose=args.verbose))
