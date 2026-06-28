#!/usr/bin/env python3
"""Convert all strain snapshot record files to VTK (mesh + GLL points).

Merges all MPI ranks, 3 source directions (x, y, z), and all time steps
into per-timestep VTK files. Includes hex cells + GLL-derived edge (LINE),
face (QUAD), and sub-volume (HEX) cells with per-point strain.

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


# ── GLL topology helpers ────────────────────────────────────────────────


def _gll_idx(i, j, k, ngll):
    return i * ngll * ngll + j * ngll + k


def build_gll_cell_template(ngll):
    """Return (edge_lines, face_quads, sub_hexes) templates (local 0-based indices)."""

    def idx(i, j, k):
        return _gll_idx(i, j, k, ngll)

    edge_lines = []
    for j, k in [(0, 0), (ngll - 1, 0), (0, ngll - 1), (ngll - 1, ngll - 1)]:
        for i in range(ngll - 1):
            edge_lines.append((idx(i, j, k), idx(i + 1, j, k)))
    for i, k in [(0, 0), (ngll - 1, 0), (0, ngll - 1), (ngll - 1, ngll - 1)]:
        for j in range(ngll - 1):
            edge_lines.append((idx(i, j, k), idx(i, j + 1, k)))
    for i, j in [(0, 0), (ngll - 1, 0), (ngll - 1, ngll - 1), (0, ngll - 1)]:
        for k in range(ngll - 1):
            edge_lines.append((idx(i, j, k), idx(i, j, k + 1)))

    face_quads = []
    for i in range(ngll - 1):
        for j in range(ngll - 1):
            face_quads.append(
                (idx(i, j, 0), idx(i + 1, j, 0), idx(i + 1, j + 1, 0), idx(i, j + 1, 0))
            )
    for i in range(ngll - 1):
        for j in range(ngll - 1):
            face_quads.append(
                (
                    idx(i, j, ngll - 1),
                    idx(i + 1, j, ngll - 1),
                    idx(i + 1, j + 1, ngll - 1),
                    idx(i, j + 1, ngll - 1),
                )
            )
    for i in range(ngll - 1):
        for k in range(ngll - 1):
            face_quads.append(
                (idx(i, 0, k), idx(i + 1, 0, k), idx(i + 1, 0, k + 1), idx(i, 0, k + 1))
            )
    for i in range(ngll - 1):
        for k in range(ngll - 1):
            face_quads.append(
                (
                    idx(i, ngll - 1, k),
                    idx(i + 1, ngll - 1, k),
                    idx(i + 1, ngll - 1, k + 1),
                    idx(i, ngll - 1, k + 1),
                )
            )
    for j in range(ngll - 1):
        for k in range(ngll - 1):
            face_quads.append(
                (idx(0, j, k), idx(0, j + 1, k), idx(0, j + 1, k + 1), idx(0, j, k + 1))
            )
    for j in range(ngll - 1):
        for k in range(ngll - 1):
            face_quads.append(
                (
                    idx(ngll - 1, j, k),
                    idx(ngll - 1, j + 1, k),
                    idx(ngll - 1, j + 1, k + 1),
                    idx(ngll - 1, j, k + 1),
                )
            )

    sub_hexes = []
    for i in range(ngll - 1):
        for j in range(ngll - 1):
            for k in range(ngll - 1):
                sub_hexes.append(
                    (
                        idx(i, j, k),
                        idx(i + 1, j, k),
                        idx(i + 1, j + 1, k),
                        idx(i, j + 1, k),
                        idx(i, j, k + 1),
                        idx(i + 1, j, k + 1),
                        idx(i + 1, j + 1, k + 1),
                        idx(i, j + 1, k + 1),
                    )
                )

    return edge_lines, face_quads, sub_hexes


def build_all_gll_cells(edge_template, face_template, sub_template, n_cell, ngll, n_mesh_vert):
    """Build full GLL cell arrays + element map for all cells.

    Returns (edge_arr, face_arr, sub_arr, n_edge, n_face, n_sub, gll_elem_map).
    """
    gll_per_cell = ngll**3

    n_edge = n_cell * len(edge_template)
    n_face = n_cell * len(face_template)
    n_sub = n_cell * len(sub_template)

    edge_arr = np.zeros(n_edge * 3, dtype=np.int32)
    face_arr = np.zeros(n_face * 5, dtype=np.int32)
    sub_arr = np.zeros(n_sub * 9, dtype=np.int32)

    for e in range(n_cell):
        base = n_mesh_vert + e * gll_per_cell
        for li, (a, b) in enumerate(edge_template):
            pos = (e * len(edge_template) + li) * 3
            edge_arr[pos] = 2
            edge_arr[pos + 1] = base + a
            edge_arr[pos + 2] = base + b
        for li, (a, b, c, d) in enumerate(face_template):
            pos = (e * len(face_template) + li) * 5
            face_arr[pos] = 4
            face_arr[pos + 1] = base + a
            face_arr[pos + 2] = base + b
            face_arr[pos + 3] = base + c
            face_arr[pos + 4] = base + d
        for li, corners in enumerate(sub_template):
            pos = (e * len(sub_template) + li) * 9
            sub_arr[pos] = 8
            for ci, corner in enumerate(corners):
                sub_arr[pos + 1 + ci] = base + corner

    gll_elem_map = np.concatenate(
        [
            np.repeat(np.arange(n_cell, dtype=np.int32), len(edge_template)),
            np.repeat(np.arange(n_cell, dtype=np.int32), len(face_template)),
            np.repeat(np.arange(n_cell, dtype=np.int32), len(sub_template)),
        ]
    )
    return edge_arr, face_arr, sub_arr, n_edge, n_face, n_sub, gll_elem_map


# ── VTK writer (legacy binary) ──────────────────────────────────────────


def write_vtu(
    path,
    vertex_coords,
    connectivity,
    cell_fields,
    point_fields,
    n_mesh_vert,
    edge_arr=None,
    face_arr=None,
    sub_arr=None,
    n_edge=0,
    n_face=0,
    n_sub=0,
    gll_elem_map=None,
):
    """Write VTK with hex cells + GLL edge/face/sub cells (binary).

    Points = [mesh_vertices | GLL_points].
    Cell ordering: [mesh_hexes | edge_LINEs | face_QUADs | sub_HEXes]
    """
    n_vert = vertex_coords.shape[0]
    n_hex = connectivity.shape[0]
    has_detail = point_fields is not None
    if has_detail:
        total_cells = n_hex + n_edge + n_face + n_sub
        total_ints = n_hex * 9 + n_edge * 3 + n_face * 5 + n_sub * 9
    else:
        total_cells = n_hex
        total_ints = n_hex * 9

    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(b"wavefield snapshot converted to VTK (detail)\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {n_vert} float\n".encode())
        f.write(np.ascontiguousarray(vertex_coords, dtype=">f4").tobytes())
        f.write(b"\n")

        f.write(f"CELLS {total_cells} {total_ints}\n".encode())
        hex_arr = np.zeros(n_hex * 9, dtype=np.int32)
        hex_arr[0::9] = 8
        for i in range(8):
            hex_arr[1 + i :: 9] = connectivity[:, i].astype(np.int32)
        if has_detail:
            cell_arr = np.concatenate([hex_arr, edge_arr, face_arr, sub_arr])
        else:
            cell_arr = hex_arr
        f.write(np.ascontiguousarray(cell_arr, dtype=">i4").tobytes())
        f.write(b"\n")

        f.write(f"CELL_TYPES {total_cells}\n".encode())
        if has_detail:
            types_arr = np.concatenate(
                [
                    np.full(n_hex, 12, dtype=np.int32),
                    np.full(n_edge, 3, dtype=np.int32),
                    np.full(n_face, 9, dtype=np.int32),
                    np.full(n_sub, 12, dtype=np.int32),
                ]
            )
        else:
            types_arr = np.full(n_hex, 12, dtype=np.int32)
        f.write(np.ascontiguousarray(types_arr, dtype=">i4").tobytes())
        f.write(b"\n")

        f.write(f"CELL_DATA {total_cells}\n".encode())
        for name, data in cell_fields.items():
            data_padded = np.zeros(total_cells, dtype=data.dtype)
            data_padded[:n_hex] = data
            if gll_elem_map is not None:
                data_padded[n_hex:] = data[gll_elem_map]
            f.write(f"SCALARS {name} float 1\n".encode())
            f.write(b"LOOKUP_TABLE default\n")
            f.write(np.ascontiguousarray(data_padded, dtype=">f4").tobytes())
            f.write(b"\n")

        if has_detail:
            f.write(f"POINT_DATA {n_vert}\n".encode())
            for name, data in point_fields.items():
                f.write(f"SCALARS {name} float 1\n".encode())
                f.write(b"LOOKUP_TABLE default\n")
                f.write(np.ascontiguousarray(data, dtype=">f4").tobytes())
                f.write(b"\n")


# ── Record file discovery ──────────────────────────────────────────────


def find_record_files(wave_dir):
    pattern = os.path.join(wave_dir, "record_*.h5")
    files = glob.glob(pattern)
    files.sort(key=lambda p: int(re.search(r"record_(\d+)\.h5$", p).group(1)))
    return files


# ── Main ───────────────────────────────────────────────────────────────


def main():
    cwd = os.getcwd()
    mesh_path = os.path.join(cwd, "mesh.h5")
    config_path = os.path.join(cwd, "config.h5")

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
        gll_coords = f["field/element/coords"][:]
    n_cell = cell_to_surface.shape[0]
    n_mesh_vert = vertex_to_coord.shape[0]
    ngll = gll_coords.shape[1]
    print(f"  Global cells: {n_cell}, vertices: {n_mesh_vert}, NGLL: {ngll}")

    print("[wavefield2vtk_detail] Resolving hexahedral connectivity...")
    connectivity = build_global_connectivity(cell_to_surface, surface_to_edge, edge_to_vertex)

    # ── Build GLL point array (reused for all timesteps) ──
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

    # ── Build GLL topology cells (reused for all timesteps) ──
    print("[wavefield2vtk_detail] Building GLL topology...")
    edge_tmpl, face_tmpl, sub_tmpl = build_gll_cell_template(ngll)
    edge_arr, face_arr, sub_arr, n_edge, n_face, n_sub, gll_elem_map = build_all_gll_cells(
        edge_tmpl, face_tmpl, sub_tmpl, n_cell, ngll, n_mesh_vert
    )
    print(
        f"  GLL topology: {n_edge}L + {n_face}Q + {n_sub}H = {n_edge + n_face + n_sub} GLL cells"
    )

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

    cell_fields_pml = {"PML_flag": is_pml.astype(np.float64)}

    out_dir = os.path.join(cwd, "vtk")
    os.makedirs(out_dir, exist_ok=True)

    # ── Iterate snapshots ──
    for snap_idx in range(n_snapshots):
        step_num = snap_idx * stride

        dir_strain = {}
        for d in _DIRECTIONS:
            gs = np.zeros((n_cell, gll_per_cell, 6), dtype=np.float64)
            for ri, f in enumerate(files[d]):
                snap = f["strain"][snap_idx]
                gs[local_eids_list[ri]] = snap.reshape(-1, gll_per_cell, 6)
            dir_strain[d] = gs

        # Build point data: 18 strain fields
        strain_field_names = [f"strain_{vl}_{d}" for d in _DIRECTIONS for vl in _VOIGT_LABELS]
        point_fields = {}
        for fi, name in enumerate(strain_field_names):
            di = fi // 6
            ci = fi % 6
            arr = np.zeros(all_points.shape[0], dtype=np.float64)
            arr[n_mesh_vert:] = dir_strain[_DIRECTIONS[di]][:, :, ci].ravel()
            point_fields[name] = arr

        out_path = os.path.join(out_dir, f"wavefield_{step_num}.vtk")
        print(f"[wavefield2vtk_detail] Writing {out_path}")
        write_vtu(
            out_path,
            all_points,
            connectivity,
            cell_fields_pml,
            point_fields,
            n_mesh_vert,
            edge_arr,
            face_arr,
            sub_arr,
            n_edge,
            n_face,
            n_sub,
            gll_elem_map,
        )

    for d in _DIRECTIONS:
        for f in files[d]:
            f.close()

    print(f"  Done. {n_snapshots} files written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
