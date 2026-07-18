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


def _build_dual_graph(topology: TopologyData) -> tuple[list[list[int]], npt.NDArray[np.int64]]:
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
    gll_coords: npt.NDArray[np.float64], n_ranks: int
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


def compute_local_cell2rank_node(
    gll_coords: npt.NDArray[np.float64], element_ids: list[int]
) -> tuple[npt.NDArray[np.int32], int]:
    """Compute per-rank local_cell2rank_node mapping from GLL coordinates (SPECFEM get_global).

    Sorts GLL node coordinates for the specified elements, then assigns
    the same per-rank global node ID (0-based) to nodes at identical
    physical positions.  Follows SPECFEM3D's ``get_global.f90`` algorithm:
    coordinate sort + distance tolerance check.

    Args:
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] — all elements' GLL coords.
        element_ids: 0-based element indices composing this rank
                     (local + ghost).  Local elements must come first.

    Returns:
        local_cell2rank_node: [n_elem, NGLL, NGLL, NGLL] int32 — per-element→node_id mapping.
        n_rank_node: number of unique per-rank global nodes.
    """
    NGLL = gll_coords.shape[1]
    n_node = NGLL * NGLL * NGLL
    n_elem = len(element_ids)

    if n_elem == 0:
        return np.zeros((0, NGLL, NGLL, NGLL), dtype=np.int32), 0

    # Subset of GLL coordinates for this rank
    rank_coords = gll_coords[element_ids]  # [n_elem, NGLL, NGLL, NGLL, 3]

    # Per-rank domain extent for floating-point tolerance
    extent = float(
        max(
            rank_coords[..., 0].max() - rank_coords[..., 0].min(),
            rank_coords[..., 1].max() - rank_coords[..., 1].min(),
            rank_coords[..., 2].max() - rank_coords[..., 2].min(),
            np.finfo(np.float64).eps,
        )
    )
    tol = np.float64(1e-12 * extent)

    n_points = n_elem * n_node

    # Flatten: 3-D coordinate columns + element/node indices
    coords_flat = rank_coords.reshape(n_points, 3)
    elem_idx = np.repeat(np.arange(n_elem, dtype=np.int32), n_node)
    node_idx = np.tile(np.arange(n_node, dtype=np.int32), n_elem)

    # Quantize coordinates onto an integer grid (scale = 1/tol) so that
    # floating-point LSB differences (~1e-13) between physically identical
    # GLL nodes on shared element faces do not split them into separate
    # sorted positions.  Sorting integers is exact (no LSB ambiguity), and
    # the adjacency diff becomes an exact equality check.  This is the
    # standard remedy for the SPECFEM get_global coordinate-sort pitfall
    # where two nodes at the same position with a 1-ULP coordinate difference
    # get separated by intervening nodes and assigned different global IDs,
    # breaking CG-SEM assembly (lost stiffness at shared nodes -> instability).
    scale = np.float64(1.0) / tol
    iquan = np.rint(coords_flat * scale).astype(np.int64)
    order = np.lexsort((iquan[:, 2], iquan[:, 1], iquan[:, 0]))
    sorted_iquan = iquan[order]
    sorted_elem = elem_idx[order]
    sorted_node = node_idx[order]

    # A new node_id starts when ANY quantized coordinate differs (exact
    # integer comparison — shared GLL nodes now cluster identically).
    is_new = np.any(np.diff(sorted_iquan, axis=0, prepend=sorted_iquan[:1] - 1) != 0, axis=1)
    rank_node_id_vals = np.cumsum(is_new, dtype=np.int32) - 1  # 0-based for C++

    # Scatter back to local_cell2rank_node[e, i, j, k]
    local_cell2rank_node = np.zeros((n_elem, NGLL, NGLL, NGLL), dtype=np.int32)
    i_idx = sorted_node // (NGLL * NGLL)
    j_idx = (sorted_node // NGLL) % NGLL
    k_idx = sorted_node % NGLL
    local_cell2rank_node[sorted_elem, i_idx, j_idx, k_idx] = rank_node_id_vals

    n_rank_node = int(rank_node_id_vals[-1]) + 1 if n_points > 0 else 0
    return local_cell2rank_node, n_rank_node


def compute_global_cell2global_node(
    gll_coords: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int32], int]:
    """Compute global element-to-node mapping from all GLL coordinates.

    One-pass coordinate sort over ALL elements (cf. the per-rank
    compute_local_cell2rank_node which operates on a subset).
    Returns a mapping usable by every rank — slicing by local element
    IDs produces each rank's ibool in a shared global numbering space.

    Args:
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] — all elements' GLL coords.

    Returns:
        global_cell2global_node: [n_cell, NGLL, NGLL, NGLL] int32
        n_global_node: number of unique global nodes.
    """
    n_cell = gll_coords.shape[0]
    NGLL = gll_coords.shape[1]
    n_node = NGLL * NGLL * NGLL

    if n_cell == 0:
        return np.zeros((0, NGLL, NGLL, NGLL), dtype=np.int32), 0

    # Global domain extent for floating-point tolerance
    extent = float(
        max(
            gll_coords[..., 0].max() - gll_coords[..., 0].min(),
            gll_coords[..., 1].max() - gll_coords[..., 1].min(),
            gll_coords[..., 2].max() - gll_coords[..., 2].min(),
            np.finfo(np.float64).eps,
        )
    )
    tol = np.float64(1e-12 * extent)

    n_points = n_cell * n_node

    # Flatten: 3-D coordinate columns + element/node indices
    coords_flat = gll_coords.reshape(n_points, 3)
    elem_idx = np.repeat(np.arange(n_cell, dtype=np.int32), n_node)
    node_idx = np.tile(np.arange(n_node, dtype=np.int32), n_cell)

    # Quantize coordinates onto an integer grid
    scale = np.float64(1.0) / tol
    iquan = np.rint(coords_flat * scale).astype(np.int64)
    order = np.lexsort((iquan[:, 2], iquan[:, 1], iquan[:, 0]))
    sorted_iquan = iquan[order]
    sorted_elem = elem_idx[order]
    sorted_node = node_idx[order]

    # A new global_node_id starts when ANY quantized coordinate differs
    is_new = np.any(np.diff(sorted_iquan, axis=0, prepend=sorted_iquan[:1] - 1) != 0, axis=1)
    global_node_id_vals = np.cumsum(is_new, dtype=np.int32) - 1  # 0-based for C++

    # Scatter back to global_cell2global_node[e, i, j, k]
    global_cell2global_node = np.zeros((n_cell, NGLL, NGLL, NGLL), dtype=np.int32)
    i_idx = sorted_node // (NGLL * NGLL)
    j_idx = (sorted_node // NGLL) % NGLL
    k_idx = sorted_node % NGLL
    global_cell2global_node[sorted_elem, i_idx, j_idx, k_idx] = global_node_id_vals

    n_global_node = int(global_node_id_vals[-1]) + 1 if n_points > 0 else 0
    return global_cell2global_node, n_global_node


def partition(topology: TopologyData, gll_coords: npt.NDArray[np.float64], n_ranks: int) -> dict:
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
          n_ranks: number of ranks
          per_rank: dict rank → dict with:
            local_cell_ids: list of 0-based element indices local to this rank
            ghost_cell_ids: list of 0-based element indices owned by other ranks
                               but needed by this rank
            ghost_owners: list of rank IDs for each ghost element
            local_cell2rank_node: [n_local_cell+n_ghost, NGLL, NGLL, NGLL] int32 — per-rank global node IDs
            n_rank_node: int — number of unique global nodes on this rank
            exchange: dict neighbor_rank → {
                "send_dof": list of per-rank global DOF indices (node_id*3+dir),
                "recv_dof": list of per-rank global DOF indices (node_id*3+dir),
            }
    """
    n_cell = topology.n_cell

    # Try METIS; if unavailable, use fallback
    try:
        import metis

        adjacency_list, _ = _build_dual_graph(topology)

        # Convert to METIS format: (start, adjacency, weight)
        if n_ranks > 1 and n_cell > 1:
            _, element_to_rank_metis = metis.part_graph(adjacency_list, n_ranks, recursive=True)
            element_to_rank = np.array(element_to_rank_metis, dtype=np.int64)
        else:
            element_to_rank = np.zeros(n_cell, dtype=np.int64)
    except ImportError:
        element_to_rank = _geometric_partition(gll_coords, n_ranks)

    # Build rank data as a dict of per-rank dicts
    per_rank: dict[int, dict] = {}
    n_cell_total = topology.n_cell
    n_surface = topology.n_surface
    c2s = topology.cell_to_surface
    NGLL = gll_coords.shape[1]
    n_node = NGLL * NGLL * NGLL

    # Precompute element → surface→face mapping for fast lookup
    # For each element e, surf_to_face[surf_idx] = face_idx (0..5)
    elem_surf_to_face: list[dict] = [{} for _ in range(n_cell_total)]
    for e in range(n_cell_total):
        for face_idx, signed_sid in enumerate(c2s[e]):
            abs_sid = abs(int(signed_sid)) - 1
            elem_surf_to_face[e][abs_sid] = face_idx

    # GLL node indices on each face
    def _face_gll_nodes(face_idx: int) -> list[int]:
        nodes = []
        if face_idx == 0:  # -z: k=0
            for i in range(NGLL):
                for j in range(NGLL):
                    nodes.append((i * NGLL + j) * NGLL + 0)
        elif face_idx == 1:  # +z: k=NGLL-1
            for i in range(NGLL):
                for j in range(NGLL):
                    nodes.append((i * NGLL + j) * NGLL + (NGLL - 1))
        elif face_idx == 2:  # -y: j=0
            for i in range(NGLL):
                for k in range(NGLL):
                    nodes.append((i * NGLL + 0) * NGLL + k)
        elif face_idx == 3:  # +y: j=NGLL-1
            for i in range(NGLL):
                for k in range(NGLL):
                    nodes.append((i * NGLL + (NGLL - 1)) * NGLL + k)
        elif face_idx == 4:  # -x: i=0
            for j in range(NGLL):
                for k in range(NGLL):
                    nodes.append((0 * NGLL + j) * NGLL + k)
        elif face_idx == 5:  # +x: i=NGLL-1
            for j in range(NGLL):
                for k in range(NGLL):
                    nodes.append(((NGLL - 1) * NGLL + j) * NGLL + k)
        return nodes

    for rank in range(n_ranks):
        locals_list: list[int] = []
        for e in range(n_cell_total):
            if element_to_rank[e] == rank:
                locals_list.append(e)
        per_rank[rank] = {
            "local_cell_ids": locals_list,
            "ghost_cell_ids": [],
            "ghost_owners": [],
            "exchange": {},
        }

    # Build surface → cells map
    surf_to_cells: dict[int, list[int]] = {}
    for e in range(n_cell_total):
        for signed_sid in c2s[e]:
            abs_sid = abs(int(signed_sid)) - 1
            if abs_sid not in surf_to_cells:
                surf_to_cells[abs_sid] = []
            if e not in surf_to_cells[abs_sid]:
                surf_to_cells[abs_sid].append(e)

    # First pass: identify ghost elements
    for surf_idx in range(n_surface):
        cells_on_surf = surf_to_cells.get(surf_idx, [])
        if len(cells_on_surf) < 2:
            continue

        for i in range(len(cells_on_surf)):
            for j in range(i + 1, len(cells_on_surf)):
                c1 = cells_on_surf[i]
                c2 = cells_on_surf[j]
                r1 = int(element_to_rank[c1])
                r2 = int(element_to_rank[c2])

                if r1 == r2:
                    continue

                for owner_rank, ghost_cell in [(r1, c2), (r2, c1)]:
                    rd = per_rank[owner_rank]
                    if ghost_cell not in rd["ghost_cell_ids"]:
                        rd["ghost_cell_ids"].append(ghost_cell)
                        rd["ghost_owners"].append(int(element_to_rank[ghost_cell]))

    # Second pass: compute exchange DOF indices
    # For CG-SEM assembly: each rank sends its local residual contributions at shared
    # interface nodes to the neighbor, and receives the neighbor's contributions at
    # the SAME local interface nodes (accumulate). send_dof == recv_dof on each rank.
    for rank in range(n_ranks):
        rd = per_rank[rank]
        local_idx_map: dict[int, int] = {
            e_global: idx for idx, e_global in enumerate(rd["local_cell_ids"])
        }
        ghost_set: set[int] = set(rd["ghost_cell_ids"])  # elements owned by other ranks
        # Build reverse map: ghost_elem → {neighbor_rank}
        ghost_to_owner: dict[int, list[int]] = {}
        for idx, g_elem in enumerate(rd["ghost_cell_ids"]):
            owner = int(rd["ghost_owners"][idx])
            if g_elem not in ghost_to_owner:
                ghost_to_owner[g_elem] = []
            ghost_to_owner[g_elem].append(owner)

        exchange_dof: dict[int, dict] = {}

        for surf_idx in range(n_surface):
            cells = surf_to_cells.get(surf_idx, [])
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    c1 = cells[i]
                    c2 = cells[j]
                    r1 = int(element_to_rank[c1])
                    r2 = int(element_to_rank[c2])

                    if r1 == r2:
                        continue

                    # Process from rank r1's perspective
                    if r1 == rank:
                        if r2 not in exchange_dof:
                            exchange_dof[r2] = {"send_dof": [], "recv_dof": []}

                        ex = exchange_dof[r2]

                        # c1 is local element → interface nodes
                        face = elem_surf_to_face[c1].get(surf_idx)
                        local_idx = local_idx_map.get(c1)
                        if face is not None and local_idx is not None:
                            for n in _face_gll_nodes(face):
                                base = local_idx * n_node * 3 + n * 3
                                for d in [base, base + 1, base + 2]:
                                    ex["send_dof"].append(d)
                                    ex["recv_dof"].append(d)

                    # Process from rank r2's perspective
                    if r2 == rank:
                        if r1 not in exchange_dof:
                            exchange_dof[r1] = {"send_dof": [], "recv_dof": []}

                        ex = exchange_dof[r1]

                        # c2 is local element → interface nodes
                        face = elem_surf_to_face[c2].get(surf_idx)
                        local_idx = local_idx_map.get(c2)
                        if face is not None and local_idx is not None:
                            for n in _face_gll_nodes(face):
                                base = local_idx * n_node * 3 + n * 3
                                for d in [base, base + 1, base + 2]:
                                    ex["send_dof"].append(d)
                                    ex["recv_dof"].append(d)

        rd["exchange"] = exchange_dof

    # Third pass: compute global element-to-node mapping once, then slice per rank.
    # Per-rank solver arrays use compact 0..n_rank_node-1 (local_cell2rank_node).
    # A parallel local_cell2global_node table (sparse global IDs) is stored for
    # read_partition_all merging across ranks.
    global_cell2global_node, n_global_node = compute_global_cell2global_node(gll_coords)

    for rank in range(n_ranks):
        rd = per_rank[rank]
        locals_list = list(rd["local_cell_ids"])
        ghosts_list = list(rd["ghost_cell_ids"])
        n_local_cell = len(locals_list)

        # Slice global mapping for this rank's elements (local + ghost).
        # Ghost elements are included so shared interface nodes receive the
        # same compact node_id within the rank.
        all_elem_ids = locals_list + ghosts_list
        ibool_global_4d = global_cell2global_node[all_elem_ids]

        # Compact relabeling: unique sorted global IDs → 0..n_rank_node-1
        _, inverse = np.unique(ibool_global_4d.ravel(), return_inverse=True)
        ibool_compact_4d = inverse.reshape(ibool_global_4d.shape).astype(np.int32)
        n_rank_node = ibool_compact_4d.max() + 1 if ibool_compact_4d.size > 0 else 0

        rd["local_cell2rank_node"] = ibool_compact_4d
        rd["local_cell2global_node"] = ibool_global_4d
        rd["n_rank_node"] = n_rank_node

        # Build map: global element ID → local index in ibool arrays
        all_elem_to_node_map = {e_global: idx for idx, e_global in enumerate(all_elem_ids)}

        # Convert exchange DOF indices: element-local → compact DOF
        for neighbor_rank, ex in rd["exchange"].items():
            for key in ("send_dof", "recv_dof"):
                old_dofs = ex[key]
                new_dofs: list[int] = []
                for old_dof in old_dofs:
                    # Decode element-local DOF:
                    #   old_dof = local_idx * n_node * 3 + node * 3 + direction
                    local_idx = old_dof // (n_node * 3)
                    remainder = old_dof % (n_node * 3)
                    node = remainder // 3
                    direction = remainder % 3

                    # Map node → (i, j, k) in GLL layout
                    k_idx = node % NGLL
                    j_idx = (node // NGLL) % NGLL
                    i_idx = node // (NGLL * NGLL)

                    node_id = int(ibool_compact_4d[local_idx, i_idx, j_idx, k_idx])
                    new_dofs.append(node_id * 3 + direction)

                ex[key] = new_dofs

            # Deduplicate exchange DOFs.  When multiple local elements on
            # this rank share interface faces with the same neighbor, the
            # shared edge/corner GLL nodes are appended once per face.
            # After conversion to rank_node IDs these become exact
            # integer duplicates.  exchange_halo does field[recv] += val
            # per entry, so duplicates cause double (or triple) counting
            # of residual and mass at shared nodes -- a major source of
            # multi-rank instability.
            seen: set[int] = set()
            uniq_send: list[int] = []
            uniq_recv: list[int] = []
            for s, r in zip(ex["send_dof"], ex["recv_dof"]):
                # send_dof == recv_dof (same node_id*3+dir) by construction
                if s not in seen:
                    seen.add(s)
                    uniq_send.append(s)
                    uniq_recv.append(r)
            ex["send_dof"] = uniq_send
            ex["recv_dof"] = uniq_recv

    return {
        "element_to_rank": element_to_rank,
        "n_ranks": n_ranks,
        "per_rank": per_rank,
        "global_cell2global_node": global_cell2global_node,
        "n_global_node": n_global_node,
    }
