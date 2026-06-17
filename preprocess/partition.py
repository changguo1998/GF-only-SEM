"""Partition elements across MPI ranks using METIS or geometric fallback.

Builds a dual graph from cell adjacency (shared surfaces), then calls
METIS k-way partitioning if available.  Falls back to simple geometric
partitioning by sorting element centroids along the longest axis.

For each rank: determines local elements, ghost elements (neighbors on
other ranks), and face-pair exchange patterns for MPI communication.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from preprocess.topology_reader import TopologyData


def _build_dual_graph(
    topology: TopologyData,
) -> tuple[list[list[int]], npt.NDArray[np.int64]]:
    """Build the dual graph of the mesh.

    Each cell is a node; edges connect cells sharing a surface.

    Returns:
        adjacency: adjacency[e] = list of neighbor cell ids.
        surf_cell_map: [n_surface] → [(cell_id, sign)] for each surface.
                       sign = +1 if the cell references the surface positively,
                       -1 if negatively.  Surfaces with 2 entries are interior
                       (shared), with 1 entry are boundary, with 0 are unused.
    """
    n_cell = topology.n_cell
    n_surface = topology.n_surface
    c2s = topology.cell_to_surface  # [n_cell, 6]

    # For each surface, track which cells reference it and with what sign
    # Mapping: surf_idx → list of (cell_idx, sign)
    surf_cell_map: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_surface)}

    for cell_idx in range(n_cell):
        for signed_sid in c2s[cell_idx]:
            abs_sid = abs(int(signed_sid)) - 1
            sign = 1 if signed_sid > 0 else -1
            surf_cell_map[abs_sid].append((cell_idx, sign))

    # Build adjacency: for shared surfaces, connect the two cells
    adjacency: list[list[int]] = [[] for _ in range(n_cell)]
    for surf_idx, cell_list in surf_cell_map.items():
        if len(cell_list) >= 2:
            # Shared surface — connect all cells sharing it
            for i in range(len(cell_list)):
                for j in range(i + 1, len(cell_list)):
                    c1 = cell_list[i][0]
                    c2 = cell_list[j][0]
                    if c2 not in adjacency[c1]:
                        adjacency[c1].append(c2)
                    if c1 not in adjacency[c2]:
                        adjacency[c2].append(c1)

    # Build surf_cell_map as a numpy-friendly structure
    scm_np = np.full(n_surface, -1, dtype=np.int64)
    for surf_idx, cell_list in surf_cell_map.items():
        if len(cell_list) >= 1:
            scm_np[surf_idx] = cell_list[0][0]

    surf_cell_arr = np.zeros((n_surface, 2), dtype=np.int64)
    surf_cell_arr.fill(-1)
    for surf_idx, cell_list in surf_cell_map.items():
        for k, (c, _) in enumerate(cell_list):
            if k < 2:
                surf_cell_arr[surf_idx, k] = c

    return adjacency, surf_cell_arr


def _geometric_partition(
    gll_coords: npt.NDArray[np.float64],
    n_ranks: int,
) -> npt.NDArray[np.int64]:
    """Fallback: partition by sorting centroids along the longest axis.

    Args:
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3]
        n_ranks: Number of partitions.

    Returns:
        element_to_rank: [n_cell] int64 — rank assignment for each element.
    """
    n_cell = gll_coords.shape[0]

    if n_ranks <= 1 or n_cell <= 1:
        return np.zeros(n_cell, dtype=np.int64)

    # Compute centroid per cell
    centroids = gll_coords.mean(axis=(1, 2, 3))  # [n_cell, 3]

    # Find longest axis
    span = np.ptp(centroids, axis=0)  # max - min per dimension
    longest_axis = int(np.argmax(span))

    # Sort by centroid along longest axis
    sort_order = np.argsort(centroids[:, longest_axis])

    # Partition into n_ranks balanced chunks
    element_to_rank = np.zeros(n_cell, dtype=np.int64)
    chunk_size = (n_cell + n_ranks - 1) // n_ranks  # ceil division

    for i, idx in enumerate(sort_order):
        rank = min(i // chunk_size, n_ranks - 1)
        element_to_rank[idx] = rank

    return element_to_rank


def partition(
    topology: TopologyData,
    gll_coords: npt.NDArray[np.float64],
    n_ranks: int,
) -> dict:
    """Partition elements across MPI ranks.

    Builds a dual graph from cell adjacency (shared surfaces), attempts
    METIS k-way partitioning, and falls back to geometric partitioning
    by centroid sorting along the longest axis.

    Args:
        topology:  Mesh topology.
        gll_coords:  GLL coords [n_cell, NGLL, NGLL, NGLL, 3] (used
                     for centroid computation in fallback).
        n_ranks:  Number of MPI ranks (partitions).

    Returns:
        dict with:
          element_to_rank: [n_cell] int64 array
          rank_data: list of dicts per rank, each with:
            local_element_ids: list of element indices local to this rank
            ghost_element_ids: list of element indices owned by other ranks
                               but needed by this rank
            ghost_owners: list of rank IDs for each ghost element
            exchange: dict neighbor_rank → list of (local_surf_idx, ghost_surf_idx)
    """
    n_cell = topology.n_cell

    # Try METIS; if unavailable, use fallback
    try:
        import metis

        adjacency_list, _ = _build_dual_graph(topology)

        # Convert to METIS format: (start, adjacency, weight)
        if n_ranks > 1 and n_cell > 1:
            _, element_to_rank_metis = metis.part_graph(
                adjacency_list, n_ranks, recursive=True
            )
            element_to_rank = np.array(element_to_rank_metis, dtype=np.int64)
        else:
            element_to_rank = np.zeros(n_cell, dtype=np.int64)
    except ImportError:
        element_to_rank = _geometric_partition(gll_coords, n_ranks)

    # Build rank data
    rank_data: list[dict] = []
    for rank in range(n_ranks):
        locals_list: list[int] = []
        for e in range(n_cell):
            if element_to_rank[e] == rank:
                locals_list.append(e)
        rank_data.append({
            "local_element_ids": locals_list,
            "ghost_element_ids": [],
            "ghost_owners": [],
            "exchange": {},
        })

    # Compute ghost elements and exchange patterns
    n_surface = topology.n_surface
    c2s = topology.cell_to_surface

    # Build surface → cells map
    surf_to_cells: dict[int, list[int]] = {}
    for e in range(n_cell):
        for signed_sid in c2s[e]:
            abs_sid = abs(int(signed_sid)) - 1
            if abs_sid not in surf_to_cells:
                surf_to_cells[abs_sid] = []
            if e not in surf_to_cells[abs_sid]:
                surf_to_cells[abs_sid].append(e)

    for surf_idx in range(n_surface):
        cells_on_surf = surf_to_cells.get(surf_idx, [])
        if len(cells_on_surf) < 2:
            continue

        # Identify pairs of cells on different ranks
        for i in range(len(cells_on_surf)):
            for j in range(i + 1, len(cells_on_surf)):
                c1 = cells_on_surf[i]
                c2 = cells_on_surf[j]
                r1 = element_to_rank[c1]
                r2 = element_to_rank[c2]

                if r1 == r2:
                    continue  # same rank, no exchange needed

                # c1 is ghost for rank r2, c2 is ghost for rank r1
                for owner_rank, ghost_cell, local_cell in [
                    (r1, c2, c1),
                    (r2, c1, c2),
                ]:
                    rd = rank_data[owner_rank]
                    if ghost_cell not in rd["ghost_element_ids"]:
                        rd["ghost_element_ids"].append(ghost_cell)
                        rd["ghost_owners"].append(element_to_rank[ghost_cell])

                    # Exchange: face pair (local_surf_idx, ghost_surf_idx)
                    # We use surface index as the face identifier
                    if element_to_rank[ghost_cell] not in rd["exchange"]:
                        rd["exchange"][int(element_to_rank[ghost_cell])] = []
                    rd["exchange"][int(element_to_rank[ghost_cell])].append(
                        (int(surf_idx), int(surf_idx))
                    )

    return {
        "element_to_rank": element_to_rank,
        "rank_data": rank_data,
    }