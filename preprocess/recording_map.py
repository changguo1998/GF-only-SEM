"""Recording map — builds shallow mesh-vertex recording map for forward solver.

Selects non-PML mesh vertices within record_depth_max_m so the forward solver
writes strain only at shallow mesh corners, not full GLL volume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from preprocess.gll_geometry import _get_cell_vertex_ids
from preprocess.topology_reader import TopologyData


def build_recording_map(
    topology: TopologyData,
    boundary_tag: npt.NDArray[np.int64],
    domain_bounds: dict[str, float],
    is_pml: npt.NDArray[np.bool_ | np.int8],
    record_depth_max_m: float,
    green_tile_size_m: float,
    element_to_rank: npt.NDArray[np.int32] | None = None,
    per_rank: dict[int, dict] | None = None,
) -> dict[str, Any]:
    """Build shallow mesh-vertex recording map.

    Args:
        topology: Mesh topology with cell→surface→edge→vertex relations.
        boundary_tag: [n_surface] int64 — 0 interior, 1 free surface, 2 absorbing.
        domain_bounds: Dict with xmin, xmax, ymin, ymax, zmin, zmax.
        is_pml: [n_cell] bool/int8 — True for PML elements.
        record_depth_max_m: Requested recording depth from surface (z upward).
        green_tile_size_m: Horizontal tile size for postprocess.
        element_to_rank: [n_cell] rank assignment (from partition).
        per_rank: Per-rank data dict from partition().

    Returns:
        Dict with:
            record_depth_actual_m: Snapped actual depth.
            per_rank_recording: dict[int, dict] — per-rank recording data:
                save_element_mask: list[bool] n_local_elem
                vertex_ids: list[int] global vertex IDs
                source_element_local_index: list[int]
                source_corner_index: list[int]
    """
    zmin = domain_bounds["zmin"]
    zmax = domain_bounds["zmax"]
    target_z = zmin + record_depth_max_m  # z positive downward

    # All elements with centroid z ≤ target_z (above or at depth limit)
    # Get vertex coords for centroid computation
    v2c = topology.vertex_to_coord  # [n_vertex, 3]

    # Build element → vertices lookup
    c2s = topology.cell_to_surface  # [n_cell, 6]
    s2e = topology.surface_to_edge  # [n_surface, 4]
    e2v = topology.edge_to_vertex  # [n_edge, 2]

    n_cell = topology.n_cell

    # Precompute element centroids
    elem_centroids = np.zeros((n_cell, 3), dtype=np.float64)
    for e in range(n_cell):
        vids = _get_cell_vertex_ids(e, c2s, s2e, e2v)
        centroid = v2c[vids - 1].mean(axis=0)
        elem_centroids[e] = centroid

    # Snap record_depth_actual_m to the nearest element face boundary
    # Find the first horizontal element face at or below target_z
    # Horizontal faces have constant z (all 4 vertices at same z)
    elem_face_z_levels = []
    for e in range(n_cell):
        vids = _get_cell_vertex_ids(e, c2s, s2e, e2v)
        z_vals = v2c[(np.array(vids, dtype=np.int64) - 1), 2]
        # Check each face (6 faces per hex)
        # Faces 0-3 are lateral, face 4 = zmin, face 5 = zmax (GMSH convention)
        for face_idx in range(6):
            # Get vertices of this face
            face_verts = _get_face_vertices(e, face_idx, c2s, s2e, e2v)
            if len(face_verts) < 4:
                continue
            face_z = v2c[face_verts, 2]
            # Horizontal face: all z equal (within tol)
            if np.max(face_z) - np.min(face_z) < 1e-12:
                z_level = float(np.mean(face_z))
                if zmin <= z_level <= zmax:
                    elem_face_z_levels.append(z_level)

    # Deduplicate and find actual depth
    unique_z = sorted(set(elem_face_z_levels))
    record_depth_actual_m = record_depth_max_m
    for z_level in unique_z:
        if z_level >= target_z:
            record_depth_actual_m = z_level - zmin
            break
    else:
        # If no face found, cap at zmax
        record_depth_actual_m = zmax - zmin

    actual_target_z = zmin + record_depth_actual_m

    # Select non-PML elements fully above depth
    selected_elems = set()
    for e in range(n_cell):
        if is_pml is not None and e < len(is_pml) and is_pml[e]:
            continue
        # Element fully above depth if its centroid z is within depth
        if elem_centroids[e, 2] <= actual_target_z + 1e-12:
            selected_elems.add(e)

    # Collect unique global vertex IDs attached to selected elements
    selected_vertex_set: set[int] = set()
    elem_vertex_map: dict[int, list[int]] = {}  # elem_idx → [8 vertex IDs]
    for e in sorted(selected_elems):
        vids = _get_cell_vertex_ids(e, c2s, s2e, e2v)
        elem_vertex_map[e] = list(vids)
        for vid in vids:
            selected_vertex_set.add(vid)

    # For each vertex, choose one owned source element and corner index
    # If no partition info, assign arbitrarily
    if element_to_rank is None or per_rank is None:
        # Single-rank fallback: assign element 0 as source for all
        per_rank_recording: dict[int, dict] = {0: _build_rank_recording(
            0, list(range(n_cell)), selected_vertex_set, elem_vertex_map, element_to_rank
        )}
    else:
        per_rank_recording = {}
        for rank, rk in per_rank.items():
            local_ids = list(rk.get("local_element_ids", []))
            local_set = set(local_ids)
            # Find which selected vertices belong to this rank
            rank_vertex_set = set()
            rank_elem_vertex_map: dict[int, list[int]] = {}
            for e in sorted(selected_elems):
                if e not in local_set:
                    continue
                vids = elem_vertex_map.get(e, [])
                rank_elem_vertex_map[e] = vids
                for vid in vids:
                    rank_vertex_set.add(vid)

            per_rank_recording[rank] = _build_rank_recording(
                rank, local_ids, rank_vertex_set, rank_elem_vertex_map, element_to_rank
            )

    return {
        "record_depth_actual_m": record_depth_actual_m,
        "per_rank_recording": per_rank_recording,
    }


def _get_face_vertices(e: int, face_idx: int, c2s, s2e, e2v) -> list[int]:
    """Get global vertex IDs for a face of element e."""
    surf_id = abs(int(c2s[e, face_idx])) - 1  # 0-based, handle signed
    if surf_id < 0:
        return []
    edge_ids = s2e[surf_id]
    verts: set[int] = set()
    for eid in edge_ids:
        eid_abs = abs(int(eid)) - 1
        if eid_abs < 0:
            continue
        v1, v2 = int(e2v[eid_abs, 0]) - 1, int(e2v[eid_abs, 1]) - 1
        verts.add(v1)
        verts.add(v2)
    return sorted(verts)


def _build_rank_recording(
    rank: int,
    local_element_ids: list[int],
    vertex_set: set[int],
    elem_vertex_map: dict[int, list[int]],
    element_to_rank: npt.NDArray[np.int32] | None,
) -> dict[str, Any]:
    """Build recording map for one rank.

    For each vertex, find a local element that contains it and assign
    the corner index.
    """
    local_set = set(local_element_ids)
    # save_element_mask: True for elements that contain at least one recorded vertex
    save_element_mask = [
        any(vid in vertex_set for vid in elem_vertex_map.get(eid, []))
        for eid in local_element_ids
    ]

    # Map vertex → (local_elem_idx, corner_idx)
    vertex_source: dict[int, tuple[int, int]] = {}

    # Build reverse: vertex → list of (local_elem_idx, corner_idx)
    vert_to_elem: dict[int, list[tuple[int, int]]] = {}
    for elem_idx, vids in elem_vertex_map.items():
        if elem_idx not in local_set:
            continue
        local_idx = local_element_ids.index(elem_idx)
        for ci, vid in enumerate(vids):
            if vid not in vertex_set:
                continue
            if vid not in vert_to_elem:
                vert_to_elem[vid] = []
            vert_to_elem[vid].append((local_idx, ci))

    for vid in sorted(vertex_set):
        sources = vert_to_elem.get(vid, [])
        if sources:
            # Pick the first (lowest local elem index, lowest corner)
            vertex_source[vid] = min(sources, key=lambda x: (x[0], x[1]))
        else:
            # Vertex not in any local element — skip
            pass

    vertex_ids = sorted(vertex_source.keys())
    source_elem_local = [vertex_source[vid][0] for vid in vertex_ids]
    source_corner = [vertex_source[vid][1] for vid in vertex_ids]

    return {
        "save_element_mask": save_element_mask,
        "vertex_ids": vertex_ids,
        "source_element_local_index": source_elem_local,
        "source_corner_index": source_corner,
    }