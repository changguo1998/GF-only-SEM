#!/usr/bin/env python3
"""Convert mesh.h5 (+ partitions/) in CWD to mesh.vtk.

Reads mesh.h5 topology and partition material fields from the current
working directory, resolves hexahedral cell connectivity, and writes
mesh.vtk with hex cells + GLL-derived edge (LINE), face (QUAD), and
sub-volume (HEX) cells with cell-averaged Vp, Vs, density, mass, PML
damping.  Viewable in ParaView / VisIt.
"""

import os

import h5py
import numpy as np

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


def build_cell_connectivity(cell_to_surface, surface_to_edge, edge_to_vertex):
    n_cell = cell_to_surface.shape[0]
    connectivity = np.zeros((n_cell, 8), dtype=np.int64)
    for ci in range(n_cell):
        conn = resolve_cell_vertices(cell_to_surface, surface_to_edge, edge_to_vertex, ci)
        connectivity[ci] = conn
    return connectivity


def read_partition_fields(partition_dir, n_cell):
    """Read cell-averaged fields from partition files.

    Returns dict of 1-D arrays [n_cell] — one scalar per element.
    """
    part_files = sorted(
        f for f in os.listdir(partition_dir) if f.startswith("partition_") and f.endswith(".h5")
    )
    if not part_files:
        raise FileNotFoundError(f"No partition_*.h5 files found in {partition_dir}")
    field_names = ["vp", "vs", "density", "mass", "damping", "tile_index"]
    fields = {name: np.zeros(n_cell, dtype=np.float64) for name in field_names}
    for pf in part_files:
        with h5py.File(os.path.join(partition_dir, pf), "r") as f:
            local_ids = f["partition/local_element_ids"][:]
            for name in field_names:
                data = f[f"field/element/{name}"][:]
                # 1-D fields (e.g. tile_index) have no GLL dims — use as-is
                if data.ndim == 1:
                    avg = data.astype(np.float64)
                else:
                    avg = np.mean(data, axis=tuple(range(1, data.ndim)))
                for li, gid in enumerate(local_ids):
                    fields[name][gid] = avg[li]
    return fields


def read_partition_gll_fields(partition_dir, n_cell, gll_shape):
    """Read raw GLL-point fields from partition files.

    Returns dict of 3-D/4-D arrays [n_cell, ngll, ngll, ngll (, comp)].
    Only material fields (vp, vs, density, mass, damping) — NOT tile_index.
    """
    part_files = sorted(
        f for f in os.listdir(partition_dir) if f.startswith("partition_") and f.endswith(".h5")
    )
    if not part_files:
        raise FileNotFoundError(f"No partition_*.h5 files found in {partition_dir}")
    field_names = ["vp", "vs", "density", "mass", "damping"]
    fields = {name: np.zeros((n_cell, *gll_shape), dtype=np.float64) for name in field_names}
    for pf in part_files:
        with h5py.File(os.path.join(partition_dir, pf), "r") as f:
            local_ids = f["partition/local_element_ids"][:]
            for name in field_names:
                data = f[f"field/element/{name}"][:].astype(np.float64)
                for li, gid in enumerate(local_ids):
                    fields[name][gid] = data[li]
    return fields


# ── GLL topology helpers ────────────────────────────────────────────────


def _gll_idx(i, j, k, ngll):
    """Flat C-order index for local GLL point (i, j, k)."""
    return i * ngll * ngll + j * ngll + k


def build_gll_cell_template(ngll):
    """Return (edge_lines, face_quads, sub_hexes) templates (local 0-based indices)."""

    def idx(i, j, k):
        return _gll_idx(i, j, k, ngll)

    edge_lines = []
    # 12 edges, (ngll-1) LINE segments each
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
    # 6 faces, (ngll-1)² QUADs each
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
    # (ngll-1)³ sub-volume HEXs
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
        f.write(b"mesh.h5 converted to VTK\n")
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


# ── Vertex-to-element helpers ───────────────────────────────────────────


def _build_vertex_to_cell_map(connectivity, n_vert):
    """Build per-vertex lists of cell indices from [n_cell, 8] connectivity."""
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
    return v2c, offsets


def _interpolate_mesh_vertex_field(cell_field, connectivity, n_vert):
    """Average cell-centered field onto mesh vertices.

    Each mesh vertex gets the mean of all elements that contain it.
    """
    v2c, offsets = _build_vertex_to_cell_map(connectivity, n_vert)
    result = np.zeros(n_vert, dtype=np.float64)
    for vi in range(n_vert):
        start = offsets[vi]
        end = offsets[vi + 1]
        if end > start:
            cells = v2c[start:end]
            result[vi] = np.mean(cell_field[cells])
    return result


# ── Main ───────────────────────────────────────────────────────────────


def main():
    cwd = os.getcwd()
    mesh_path = os.path.join(cwd, "mesh.h5")
    vtk_dir = os.path.join(cwd, "vtk")
    out_path = os.path.join(vtk_dir, "mesh.vtk")
    part_dir = os.path.join(cwd, "partitions")

    has_partitions = os.path.isdir(part_dir)

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

    print("[mesh_to_vtk] Resolving hexahedral connectivity...")
    connectivity = build_cell_connectivity(cell_to_surface, surface_to_edge, edge_to_vertex)

    cell_fields = {}
    point_fields = None
    n_mesh_vert = None
    vertex_coords_out = vertex_to_coord
    edge_arr = face_arr = sub_arr = None
    n_edge = n_face = n_sub = 0
    gll_elem_map = None

    if has_partitions:
        ngll_dim = None
        # Peek at GLL dims if available (for reading raw GLL fields)
        if gll_coords is not None:
            ngll_dim = gll_coords.shape[1]

        print("[mesh_to_vtk] Reading partitions/...")
        fields = read_partition_fields(part_dir, n_cell)

        # ── Cell-root fields (stay as CELL_DATA, no broadcast to points) ──
        cell_fields["PML_flag"] = is_pml.astype(np.float64)
        cell_fields["Tile_Index"] = fields.get("tile_index", np.full(n_cell, -1.0))
        print("  Cell fields: " + ", ".join(cell_fields.keys()))

        # ── Point-root fields (stay as POINT_DATA, no averaging to cells) ──
        if ngll_dim is not None and gll_coords is not None:
            gll_fields = read_partition_gll_fields(
                part_dir, n_cell, (ngll_dim, ngll_dim, ngll_dim)
            )
            ngll = ngll_dim
            gll_per_cell = ngll**3
            n_mesh_vert = n_vert

            gll_pt_list = []
            for ci in range(n_cell):
                gll_pt_list.append(gll_coords[ci].reshape(-1, 3))
            gll_points = np.concatenate(gll_pt_list, axis=0) if gll_pt_list else np.empty((0, 3))
            vertex_coords_out = np.concatenate([vertex_to_coord, gll_points], axis=0)

            print("[mesh_to_vtk] Building GLL point data and topology...")
            # Point fields from raw GLL data — no cell averaging, no broadcast
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
                # Use cell-averaged value as estimator for mesh vertex locations
                cell_avg = np.mean(gll_fields[name_raw].reshape(n_cell, -1), axis=1)
                arr[:n_mesh_vert] = _interpolate_mesh_vertex_field(
                    cell_avg, connectivity, n_mesh_vert
                )
                # GLL points: use raw values directly (not cell-averaged)
                for ci in range(n_cell):
                    s = n_mesh_vert + ci * gll_per_cell
                    arr[s : s + gll_per_cell] = gll_fields[name_raw][ci].ravel()
                point_fields[name_h5] = arr
            print("  Point fields: " + ", ".join(point_fields.keys()))

            # Build GLL topology for detail view
            edge_tmpl, face_tmpl, sub_tmpl = build_gll_cell_template(ngll)
            edge_arr, face_arr, sub_arr, n_edge, n_face, n_sub, gll_elem_map = build_all_gll_cells(
                edge_tmpl, face_tmpl, sub_tmpl, n_cell, ngll, n_mesh_vert
            )
            print(f"  GLL per cell: {gll_per_cell}, total GLL: {n_cell * gll_per_cell}")
            print(
                f"  Topology: {n_edge}L + {n_face}Q + {n_sub}H = "
                f"{n_edge + n_face + n_sub} GLL cells"
            )
    else:
        # No partitions — PML_flag and Tile_Index only as cell fields
        cell_fields["PML_flag"] = is_pml.astype(np.float64)
        cell_fields["Tile_Index"] = np.full(n_cell, -1.0)
        print("  Fields: PML_flag only (no partitions/)")

        edge_tmpl, face_tmpl, sub_tmpl = build_gll_cell_template(ngll)
        edge_arr, face_arr, sub_arr, n_edge, n_face, n_sub, gll_elem_map = build_all_gll_cells(
            edge_tmpl, face_tmpl, sub_tmpl, n_cell, ngll, n_mesh_vert
        )
        print(f"  GLL per cell: {gll_per_cell}, total GLL: {n_cell * gll_per_cell}")
        print(
            f"  Topology: {n_edge}L + {n_face}Q + {n_sub}H = {n_edge + n_face + n_sub} GLL cells"
        )

    os.makedirs(vtk_dir, exist_ok=True)
    print(f"[mesh_to_vtk] Writing {out_path}")
    write_vtu(
        out_path,
        vertex_coords_out,
        connectivity,
        cell_fields,
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
    print("  Done.")


if __name__ == "__main__":
    main()
