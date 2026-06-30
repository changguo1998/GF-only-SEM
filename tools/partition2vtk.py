#!/usr/bin/env python3
"""Convert partitioned mesh (partitions/partition_{r}.h5) to per-rank VTK files.

Reads model.h5 topology and each partition file from the partitions/ directory
in CWD, resolves hexahedral cell connectivity for local elements, and writes
partition_{r}.vtk per rank with cell-averaged Vp, Vs, density, mass, damping,
PML flag, and rank assignment.

Uses GLL-derived edge (LINE), face (QUAD), and sub-volume (HEX) cells for
proper ParaView interpolation.

Useful for visually inspecting METIS partition quality and per-rank element
distribution in ParaView / VisIt.
"""

import os
import argparse
import re

import h5py
import numpy as np


def _interpolate_local_vertex_field(cell_field, connectivity, n_vert):
    """Average cell-centered field onto mesh vertices (local connectivity)."""
    counts = np.zeros(n_vert, dtype=np.int32)
    for ci in range(connectivity.shape[0]):
        for v in connectivity[ci]:
            counts[v] += 1
    offsets = np.zeros(n_vert + 1, dtype=np.int32)
    np.cumsum(counts, out=offsets[1:])
    v2c = np.zeros(offsets[-1], dtype=np.int32)
    cur = np.zeros(n_vert, dtype=np.int32)
    for ci in range(connectivity.shape[0]):
        for v in connectivity[ci]:
            pos = offsets[v] + cur[v]
            v2c[pos] = ci
            cur[v] += 1
    result = np.zeros(n_vert, dtype=np.float64)
    for vi in range(n_vert):
        start = offsets[vi]
        end = offsets[vi + 1]
        if end > start:
            cells = v2c[start:end]
            result[vi] = np.mean(cell_field[cells])
    return result


_HEX_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5]]


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


def build_local_connectivity(local_element_ids, cell_to_surface, surface_to_edge, edge_to_vertex):
    """Build [n_local, 8] connectivity for local elements (0-based vertex indices)."""
    n_local = len(local_element_ids)
    connectivity = np.zeros((n_local, 8), dtype=np.int64)
    for li, gid in enumerate(local_element_ids):
        conn = resolve_cell_vertices(cell_to_surface, surface_to_edge, edge_to_vertex, gid)
        connectivity[li] = conn
    return connectivity


# ── GLL topology helpers ────────────────────────────────────────────────


def _gll_idx(i, j, k, ngll):
    """Flat C-order index for local GLL point (i, j, k)."""
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


def build_all_gll_cells(edge_template, face_template, sub_template, n_local, ngll, n_mesh_vert):
    """Build full GLL cell arrays + element map for local cells.

    Returns (edge_arr, face_arr, sub_arr, n_edge, n_face, n_sub, gll_elem_map).
    """
    gll_per_cell = ngll**3

    n_edge = n_local * len(edge_template)
    n_face = n_local * len(face_template)
    n_sub = n_local * len(sub_template)

    edge_arr = np.zeros(n_edge * 3, dtype=np.int32)
    face_arr = np.zeros(n_face * 5, dtype=np.int32)
    sub_arr = np.zeros(n_sub * 9, dtype=np.int32)

    for li in range(n_local):
        base = n_mesh_vert + li * gll_per_cell
        for tli, (a, b) in enumerate(edge_template):
            pos = (li * len(edge_template) + tli) * 3
            edge_arr[pos] = 2
            edge_arr[pos + 1] = base + a
            edge_arr[pos + 2] = base + b
        for tli, (a, b, c, d) in enumerate(face_template):
            pos = (li * len(face_template) + tli) * 5
            face_arr[pos] = 4
            face_arr[pos + 1] = base + a
            face_arr[pos + 2] = base + b
            face_arr[pos + 3] = base + c
            face_arr[pos + 4] = base + d
        for tli, corners in enumerate(sub_template):
            pos = (li * len(sub_template) + tli) * 9
            sub_arr[pos] = 8
            for ci, corner in enumerate(corners):
                sub_arr[pos + 1 + ci] = base + corner

    gll_elem_map = np.concatenate(
        [
            np.repeat(np.arange(n_local, dtype=np.int32), len(edge_template)),
            np.repeat(np.arange(n_local, dtype=np.int32), len(face_template)),
            np.repeat(np.arange(n_local, dtype=np.int32), len(sub_template)),
        ]
    )
    return edge_arr, face_arr, sub_arr, n_edge, n_face, n_sub, gll_elem_map


# ── VTK writer (legacy binary) ──────────────────────────────────────────


def write_vtu(
    path,
    vertex_coords,
    connectivity,
    cell_fields,
    point_fields=None,
    n_mesh_vert=None,
    edge_arr=None,
    face_arr=None,
    sub_arr=None,
    n_edge=0,
    n_face=0,
    n_sub=0,
    gll_elem_map=None,
):
    """Write VTK with mesh hex cells + GLL edge/face/sub cells (binary).

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
        f.write(b"partition file converted to VTK\n")
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


# ── Main ───────────────────────────────────────────────────────────────


def main(verbose: bool = False):
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "model.h5")
    part_dir = os.path.join(cwd, "partitions")
    vtk_dir = os.path.join(cwd, "vtk")
    if not os.path.isdir(part_dir):
        print("[partition_to_vtk] Error: no partitions/ directory found in CWD")
        return 1

    print(f"[partition_to_vtk] Reading {model_path}")
    with h5py.File(model_path, "r") as f:
        topo = f["topology"]
        vertex_to_coord = topo["vertex_to_coord"][:]
        edge_to_vertex = topo["edge_to_vertex"][:]
        surface_to_edge = topo["surface_to_edge"][:]
        cell_to_surface = topo["cell_to_surface"][:]

        is_pml_global = np.zeros(cell_to_surface.shape[0], dtype=np.int8)
        if "field/element/is_pml" in f:
            is_pml_global[:] = f["field/element/is_pml"][:]

        gll_coords = f["field/element/coords"][:] if "field/element/coords" in f else None

    n_cell = cell_to_surface.shape[0]
    n_vert = vertex_to_coord.shape[0]
    print(f"  Global cells: {n_cell}, vertices: {n_vert}")

    part_files = sorted(
        f for f in os.listdir(part_dir) if f.startswith("partition_") and f.endswith(".h5")
    )
    if not part_files:
        print("[partition_to_vtk] Error: no partition_*.h5 files in partitions/")
        return 1

    if verbose:
        print(f"[partition_to_vtk] Found {len(part_files)} partition files")

    if gll_coords is not None:
        ngll = gll_coords.shape[1]
        edge_tmpl, face_tmpl, sub_tmpl = build_gll_cell_template(ngll)
    else:
        ngll = 0
        edge_tmpl = face_tmpl = sub_tmpl = None

    os.makedirs(vtk_dir, exist_ok=True)

    for pf in part_files:
        m = re.match(r"partition_(\d+)\.h5$", pf)
        if not m:
            continue
        rank = int(m.group(1))
        part_path = os.path.join(part_dir, pf)
        out_path = os.path.join(vtk_dir, f"partition_{rank}.vtk")

        with h5py.File(part_path, "r") as f:
            local_zero = f["partition/local_element_ids"][:]
            n_local = len(local_zero)

            connectivity = build_local_connectivity(
                local_zero, cell_to_surface, surface_to_edge, edge_to_vertex
            )

            # ── Cell-root fields (read inside with block) ──
            tile_raw = f["field/element/tile_index"][:].astype(np.float64)

            # ── Point-root raw GLL data (read inside with block) ──
            raw_gll = {}
            for name in ["vp", "vs", "density", "mass", "damping"]:
                raw_gll[name] = f[f"field/element/{name}"][:].astype(np.float64)

        vtk_fields = {
            "PML_flag": is_pml_global[local_zero].astype(np.float64),
            "Rank": np.full(n_local, float(rank)),
            "Tile_Index": tile_raw if tile_raw.ndim == 1 else np.full(n_local, -1.0),
        }

        # ── Point-root fields (POINT_DATA only, from raw GLL data) ──
        point_fields = None
        n_mesh_vert = None
        vertex_coords_out = vertex_to_coord
        edge_arr = face_arr = sub_arr = None
        n_edge = n_face = n_sub = 0
        gll_elem_map = None

        if gll_coords is not None:
            gll_per_cell = ngll**3
            n_mesh_vert = n_vert

            gll_pt_list = []
            for ci in local_zero:
                gll_pt_list.append(gll_coords[ci].reshape(-1, 3))
            gll_points = np.concatenate(gll_pt_list, axis=0) if gll_pt_list else np.empty((0, 3))
            vertex_coords_out = np.concatenate([vertex_to_coord, gll_points], axis=0)

            point_fields = {}
            for name_h5, name_raw in [
                ("Vp_m_s", "vp"),
                ("Vs_m_s", "vs"),
                ("Density_kg_m3", "density"),
                ("Mass", "mass"),
                ("PML_Damping", "damping"),
            ]:
                arr = np.zeros(vertex_coords_out.shape[0], dtype=np.float64)
                # Mesh vertices: interpolate from surrounding elements (point→point)
                cell_avg = np.mean(raw_gll[name_raw].reshape(n_local, -1), axis=1)
                arr[:n_mesh_vert] = _interpolate_local_vertex_field(
                    cell_avg, connectivity, n_mesh_vert
                )
                # GLL points: use raw values directly
                for li in range(n_local):
                    s = n_mesh_vert + li * gll_per_cell
                    arr[s : s + gll_per_cell] = raw_gll[name_raw][li].ravel()
                point_fields[name_h5] = arr

            edge_arr, face_arr, sub_arr, n_edge, n_face, n_sub, gll_elem_map = build_all_gll_cells(
                edge_tmpl, face_tmpl, sub_tmpl, n_local, ngll, n_mesh_vert
            )

        if verbose:
            print(f"  [{pf}] {n_local} cells → {out_path}")
        write_vtu(
            out_path,
            vertex_coords_out,
            connectivity,
            vtk_fields,
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

    print("[partition_to_vtk] Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert partition files to per-rank VTK files.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed processing messages"
    )
    args = parser.parse_args()
    raise SystemExit(main(verbose=args.verbose))
