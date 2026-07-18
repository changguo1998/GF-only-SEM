"""Recording map — builds shallow mesh-vertex recording map for forward solver.

Selects non-PML mesh vertices within record_depth_max_m so the forward solver
writes strain only at shallow mesh corners, not full GLL volume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from preprocess.gll_geometry import HEX_REF_CORNERS, _get_cell_vertex_ids
from preprocess.topology_reader import TopologyData

# The forward solver records displacement at element corner GLL nodes using a
# 3-bit corner index: bit 0 -> i = NGLL-1, bit 1 -> j = NGLL-1, bit 2 -> k = NGLL-1
# (see forward/share/src/solver.cpp RecordWriter usage). _get_cell_vertex_ids
# returns vertices in GMSH hex order (HEX_REF_CORNERS), whose (xi, eta, zeta) signs
# differ from the bit-flag order at positions 2<->3 and 6<->7. Convert the GMSH
# positional index to the solver bit-flag corner so each recorded vertex maps to
# the GLL corner node it actually coincides with.
_GMSH_CORNER_TO_BITFLAG = [
    int((c[0] > 0) | ((c[1] > 0) << 1) | ((c[2] > 0) << 2)) for c in HEX_REF_CORNERS
]


def build_recording_map(
    topology: TopologyData,
    domain_bounds: dict[str, float],
    is_pml: npt.NDArray[np.bool_ | np.int8],
    record_depth_max_m: float,
    element_to_rank: npt.NDArray[np.int32] | None = None,
    per_rank: dict[int, dict] | None = None,
) -> dict[str, Any]:
    """Build shallow mesh-vertex recording map.

    Args:
        topology: Mesh topology with cell→surface→edge→vertex relations.
        domain_bounds: Dict with xmin, xmax, ymin, ymax, zmin, zmax.
        is_pml: [n_cell] bool/int8 — True for PML elements.
        record_depth_max_m: Requested recording depth from surface (z upward).
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
    vertex_coords = topology.vertex_to_coord  # [n_vertex, 3]

    # Build element → vertices lookup
    cell_to_surface = topology.cell_to_surface  # [n_cell, 6]
    surface_to_edge = topology.surface_to_edge  # [n_surface, 4]
    edge_to_vertex = topology.edge_to_vertex  # [n_edge, 2]

    n_cell = topology.n_cell

    # Precompute element centroids
    elem_centroids = np.zeros((n_cell, 3), dtype=np.float64)
    for elem in range(n_cell):
        vertex_ids = _get_cell_vertex_ids(elem, cell_to_surface, surface_to_edge, edge_to_vertex)
        centroid = vertex_coords[vertex_ids - 1].mean(axis=0)
        elem_centroids[elem] = centroid

    # Snap record_depth_actual_m to the nearest element face boundary
    # Find the first horizontal element face at or below target_z
    # Horizontal faces have constant z (all 4 vertices at same z)
    elem_face_z_levels = []
    for elem in range(n_cell):
        vertex_ids = _get_cell_vertex_ids(elem, cell_to_surface, surface_to_edge, edge_to_vertex)
        z_vals = vertex_coords[(np.array(vertex_ids, dtype=np.int64) - 1), 2]
        # Check each face (6 faces per hex)
        # Faces 0-3 are lateral, face 4 = zmin, face 5 = zmax (GMSH convention)
        for face_idx in range(6):
            # Get vertices of this face
            face_verts = _get_face_vertices(
                elem, face_idx, cell_to_surface, surface_to_edge, edge_to_vertex
            )
            if len(face_verts) < 4:
                continue
            face_z = vertex_coords[face_verts, 2]
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
    for elem in range(n_cell):
        if is_pml is not None and elem < len(is_pml) and is_pml[elem]:
            continue
        # Element fully above depth if its centroid z is within depth
        if elem_centroids[elem, 2] <= actual_target_z + 1e-12:
            selected_elems.add(elem)

    # Collect unique global vertex IDs attached to selected elements
    selected_vertex_set: set[int] = set()
    elem_vertex_map: dict[int, list[int]] = {}  # elem_idx → [8 vertex IDs]
    for elem in sorted(selected_elems):
        vertex_ids = _get_cell_vertex_ids(elem, cell_to_surface, surface_to_edge, edge_to_vertex)
        elem_vertex_map[elem] = list(vertex_ids)
        for global_vertex_id in vertex_ids:
            selected_vertex_set.add(global_vertex_id)

    # For each vertex, choose one owned source element and corner index
    # If no partition info, assign arbitrarily
    if element_to_rank is None or per_rank is None:
        # Single-rank fallback: assign element 0 as source for all
        per_rank_recording: dict[int, dict] = {
            0: _build_rank_recording(0, list(range(n_cell)), selected_vertex_set, elem_vertex_map)
        }
    else:
        per_rank_recording = {}
        for rank, rank_data in per_rank.items():
            local_ids = list(rank_data.get("local_cell_ids", []))
            local_set = set(local_ids)
            # Find which selected vertices belong to this rank
            rank_vertex_set = set()
            rank_elem_vertex_map: dict[int, list[int]] = {}
            for elem in sorted(selected_elems):
                if elem not in local_set:
                    continue
                vertex_ids = elem_vertex_map.get(elem, [])
                rank_elem_vertex_map[elem] = vertex_ids
                for global_vertex_id in vertex_ids:
                    rank_vertex_set.add(global_vertex_id)

            per_rank_recording[rank] = _build_rank_recording(
                rank, local_ids, rank_vertex_set, rank_elem_vertex_map
            )

    return {
        "record_depth_actual_m": record_depth_actual_m,
        "per_rank_recording": per_rank_recording,
    }


def _get_face_vertices(
    elem: int, face_idx: int, cell_to_surface, surface_to_edge, edge_to_vertex
) -> list[int]:
    """Get 0-based vertex array indices for a face of element elem.

    Returns 0-based indices into the vertex coordinate array (i.e. global
    vertex ID minus 1), suitable for direct indexing of vertex_to_coord.
    """
    surf_id = abs(int(cell_to_surface[elem, face_idx])) - 1  # 0-based, handle signed
    if surf_id < 0:
        return []
    edge_ids = surface_to_edge[surf_id]
    verts: set[int] = set()
    for signed_edge_id in edge_ids:
        edge_index = abs(int(signed_edge_id)) - 1
        if edge_index < 0:
            continue
        vertex_a, vertex_b = (
            int(edge_to_vertex[edge_index, 0]) - 1,
            int(edge_to_vertex[edge_index, 1]) - 1,
        )
        verts.add(vertex_a)
        verts.add(vertex_b)
    return sorted(verts)


def _build_rank_recording(
    rank: int,
    local_cell_ids: list[int],
    vertex_set: set[int],
    elem_vertex_map: dict[int, list[int]],
) -> dict[str, Any]:
    """Build recording map for one rank.

    For each vertex, find a local element that contains it and assign
    the corner index.
    """
    local_set = set(local_cell_ids)
    # save_element_mask: True for elements that contain at least one recorded vertex
    save_element_mask = [
        any(
            global_vertex_id in vertex_set for global_vertex_id in elem_vertex_map.get(elem_id, [])
        )
        for elem_id in local_cell_ids
    ]

    # Map vertex → (local_elem_idx, corner_idx)
    vertex_source: dict[int, tuple[int, int]] = {}

    # Pre-build element → local_index mapping (O(n) instead of O(n²))
    element_to_local_index: dict[int, int] = {
        elem_id: idx for idx, elem_id in enumerate(local_cell_ids)
    }

    # Build reverse: vertex → list of (local_elem_idx, corner_idx)
    vert_to_elem: dict[int, list[tuple[int, int]]] = {}
    for elem_idx, vertex_ids in elem_vertex_map.items():
        if elem_idx not in local_set:
            continue
        local_idx = element_to_local_index[elem_idx]
        for gmsh_pos, global_vertex_id in enumerate(vertex_ids):
            corner_index = _GMSH_CORNER_TO_BITFLAG[gmsh_pos]
            if global_vertex_id not in vertex_set:
                continue
            if global_vertex_id not in vert_to_elem:
                vert_to_elem[global_vertex_id] = []
            vert_to_elem[global_vertex_id].append((local_idx, corner_index))

    for global_vertex_id in sorted(vertex_set):
        sources = vert_to_elem.get(global_vertex_id, [])
        if sources:
            # Pick the first (lowest local elem index, lowest corner)
            vertex_source[global_vertex_id] = min(
                sources, key=lambda source_pair: (source_pair[0], source_pair[1])
            )
        else:
            # Vertex not in any local element — skip
            pass

    vertex_ids = sorted(vertex_source.keys())
    source_elem_local = [vertex_source[global_vertex_id][0] for global_vertex_id in vertex_ids]
    source_corner = [vertex_source[global_vertex_id][1] for global_vertex_id in vertex_ids]

    return {
        "save_element_mask": save_element_mask,
        "vertex_ids": vertex_ids,
        "source_element_local_index": source_elem_local,
        "source_corner_index": source_corner,
    }
