"""Recording map — builds GLL-node recording map for forward solver.

Selects all NGLL^3 GLL nodes per recording-region cell (within record_depth_max_m
of free surface, excluding PML). Deduplicates shared GLL nodes across cells to
produce a unique global node list for continuous-output tile storage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from preprocess.topology_reader import TopologyData


def build_recording_map(
    topology: TopologyData,
    domain_bounds: dict[str, float],
    is_pml: npt.NDArray[np.bool_ | np.int8],
    record_depth_max_m: float,
    global_cell2global_node: npt.NDArray[np.int64 | np.int32],
    gll_node_coords: npt.NDArray[np.float64],
    element_to_rank: npt.NDArray[np.int32] | None = None,
    per_rank: dict[int, dict] | None = None,
) -> dict[str, Any]:
    """Build GLL-node recording map.

    Args:
        topology: Mesh topology (used for n_cell count only).
        domain_bounds: Dict with xmin, xmax, ymin, ymax, zmin, zmax.
        is_pml: [n_cell] bool/int8 — True for PML cells.
        record_depth_max_m: Requested recording depth from surface (z downward).
        global_cell2global_node: [n_cell, NGLL, NGLL, NGLL] global GLL node IDs.
        gll_node_coords: [n_cell, NGLL, NGLL, NGLL, 3] physical coordinates.
        element_to_rank: [n_cell] rank assignment (from partition).
        per_rank: Per-rank data dict from partition().

    Returns:
        Dict with:
            record_depth_actual_m: Snapped actual depth.
            per_rank_recording: dict[int, dict] — per-rank recording data:
                rec_cell_global_ids: list[int] recording cell global IDs
                rec_cell_local_index: list[int] local index within rank
                cell_gll_node_ids: list[list[int]] per-cell NGLL^3 GLL node IDs
                gll_node_ids: list[int] unique global GLL node IDs
                gll_node_coords: list[list[float]] coordinates [n_unique, 3]
                cell_gll_node_index: list[list[int]] index into gll_node_ids
    """
    zmin = domain_bounds["zmin"]
    zmax = domain_bounds["zmax"]
    target_z = zmin + record_depth_max_m  # z positive downward

    n_cell = topology.n_cell
    ngll = global_cell2global_node.shape[1]  # NGLL = N+1

    # Precompute cell centroids and z-levels from GLL node coordinates
    gll_coords_flat = gll_node_coords.reshape(n_cell, ngll * ngll * ngll, 3)
    cell_centroids = gll_coords_flat.mean(axis=1)  # [n_cell, 3]

    # Snap record_depth_actual_m to the nearest cell z-boundary
    elem_face_z_levels: list[float] = []
    for cell in range(n_cell):
        z_vals = gll_coords_flat[cell, :, 2]
        z_min_cell = float(z_vals.min())
        z_max_cell = float(z_vals.max())
        if zmin <= z_min_cell <= zmax:
            elem_face_z_levels.append(z_min_cell)
        if zmin <= z_max_cell <= zmax:
            elem_face_z_levels.append(z_max_cell)

    unique_z = sorted(set(elem_face_z_levels))
    record_depth_actual_m = record_depth_max_m
    for z_level in unique_z:
        if z_level >= target_z:
            record_depth_actual_m = z_level - zmin
            break
    else:
        record_depth_actual_m = zmax - zmin

    actual_target_z = zmin + record_depth_actual_m

    # Select non-PML cells within recording depth
    selected_cells: set[int] = set()
    for cell in range(n_cell):
        if is_pml is not None and cell < len(is_pml) and is_pml[cell]:
            continue
        if cell_centroids[cell, 2] <= actual_target_z + 1e-12:
            selected_cells.add(cell)

    # Build per-rank recording maps
    if element_to_rank is None or per_rank is None:
        per_rank_recording: dict[int, dict] = {
            0: _build_rank_recording(
                0,
                list(range(n_cell)),
                selected_cells,
                global_cell2global_node,
                gll_node_coords,
                ngll,
            )
        }
    else:
        per_rank_recording = {}
        for rank, rank_data in per_rank.items():
            local_ids = list(rank_data.get("local_cell_ids", []))
            per_rank_recording[rank] = _build_rank_recording(
                rank, local_ids, selected_cells, global_cell2global_node, gll_node_coords, ngll
            )

    return {
        "record_depth_actual_m": record_depth_actual_m,
        "per_rank_recording": per_rank_recording,
    }


def _build_rank_recording(
    rank: int,
    local_cell_ids: list[int],
    recording_cell_set: set[int],
    global_cell2global_node: npt.NDArray[np.int64 | np.int32],
    gll_node_coords_all: npt.NDArray[np.float64],
    ngll: int,
) -> dict[str, Any]:
    """Build GLL-node recording map for one rank.

    Selects all NGLL^3 GLL nodes per recording cell. Deduplicates shared nodes
    across cells to produce a unique global node list.
    """
    # Select recording cells that are local to this rank
    rec_cell_local = [idx for idx, cid in enumerate(local_cell_ids) if cid in recording_cell_set]
    rec_cell_global = [local_cell_ids[idx] for idx in rec_cell_local]

    # Collect GLL node IDs for each recording cell
    cell_gll_node_ids: list[list[int]] = []
    all_node_ids: set[int] = set()
    for gcell in rec_cell_global:
        nodes = global_cell2global_node[gcell].ravel().tolist()  # [n_node]
        cell_gll_node_ids.append(nodes)
        all_node_ids.update(nodes)

    # Deduplicate: unique global GLL node IDs
    gll_node_ids = sorted(all_node_ids)
    node_id_to_index = {nid: idx for idx, nid in enumerate(gll_node_ids)}

    # cell_gll_node_index: [n_rec_cell, n_node] -> index into gll_node_ids
    cell_gll_node_index = [[node_id_to_index[n] for n in nodes] for nodes in cell_gll_node_ids]

    # gll_node_coords: [n_unique_gll, 3]
    # Build global node ID -> coord map from first occurrence in any cell
    node_coord_map: dict[int, list[float]] = {}
    for ci, gcell in enumerate(rec_cell_global):
        coords_cell = gll_node_coords_all[gcell]  # [ngll, ngll, ngll, 3]
        for i in range(ngll):
            for j in range(ngll):
                for k in range(ngll):
                    nid = cell_gll_node_ids[ci][i * ngll * ngll + j * ngll + k]
                    if nid not in node_coord_map:
                        node_coord_map[nid] = coords_cell[i, j, k].tolist()
    gll_node_coords = [node_coord_map[nid] for nid in gll_node_ids]

    return {
        "rec_cell_global_ids": rec_cell_global,
        "rec_cell_local_index": rec_cell_local,
        "cell_gll_node_ids": cell_gll_node_ids,
        "gll_node_ids": gll_node_ids,
        "gll_node_coords": gll_node_coords,
        "cell_gll_node_index": cell_gll_node_index,
    }
