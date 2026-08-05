"""Partition elements across MPI ranks using METIS (topology only).

Builds a dual graph from cell adjacency (shared surfaces), calls
METIS k-way partitioning, then computes exchange DOF patterns and
global node numbering from topology face adjacency — no coordinate
comparison needed.

For each rank: determines local elements, ghost elements (neighbors on
other ranks), and face-pair exchange patterns for MPI communication.
"""

from __future__ import annotations

import pymetis  # required — bundles native METIS, no system lib needed
import numpy as np
import numpy.typing as npt

from preprocess.topology_reader import TopologyData


def _build_dual_graph(topology: TopologyData) -> tuple[list[list[int]], npt.NDArray[np.int64]]:
    """Build the dual graph of the mesh.

    Each cell is a node; edges connect cells sharing a surface.

    Returns:
        adjacency: adjacency[e] = list of neighbor cell ids.
        surf_cell_arr: [n_surface, 2] — up to two cells per surface (-1 = empty).
    """
    n_cell = topology.n_cell
    n_surface = topology.n_surface
    c2s = topology.cell_to_surface

    surf_cell_map: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_surface)}

    for cell_idx in range(n_cell):
        for signed_sid in c2s[cell_idx]:
            abs_sid = abs(int(signed_sid)) - 1
            sign = 1 if signed_sid > 0 else -1
            surf_cell_map[abs_sid].append((cell_idx, sign))

    adjacency: list[list[int]] = [[] for _ in range(n_cell)]
    for surf_idx, cell_list in surf_cell_map.items():
        if len(cell_list) >= 2:
            for i in range(len(cell_list)):
                for j in range(i + 1, len(cell_list)):
                    c1 = cell_list[i][0]
                    c2 = cell_list[j][0]
                    if c2 not in adjacency[c1]:
                        adjacency[c1].append(c2)
                    if c1 not in adjacency[c2]:
                        adjacency[c2].append(c1)

    surf_cell_arr = np.full((n_surface, 2), -1, dtype=np.int64)
    for surf_idx, cell_list in surf_cell_map.items():
        for k, (c, _) in enumerate(cell_list):
            if k < 2:
                surf_cell_arr[surf_idx, k] = c

    return adjacency, surf_cell_arr


# ── Face GLL node helpers ─────────────────────────────────────────────────


def _face_gll_nodes(face_idx: int, ngll: int) -> list[int]:
    """Flat node indices (0..ngll³-1) on the specified face."""
    nodes: list[int] = []
    if face_idx == 0:  # -z: k=0
        for i in range(ngll):
            for j in range(ngll):
                nodes.append((i * ngll + j) * ngll + 0)
    elif face_idx == 1:  # +z: k=ngll-1
        for i in range(ngll):
            for j in range(ngll):
                nodes.append((i * ngll + j) * ngll + (ngll - 1))
    elif face_idx == 2:  # -y: j=0
        for i in range(ngll):
            for k in range(ngll):
                nodes.append((i * ngll + 0) * ngll + k)
    elif face_idx == 3:  # +y: j=ngll-1
        for i in range(ngll):
            for k in range(ngll):
                nodes.append((i * ngll + (ngll - 1)) * ngll + k)
    elif face_idx == 4:  # -x: i=0
        for j in range(ngll):
            for k in range(ngll):
                nodes.append((0 * ngll + j) * ngll + k)
    elif face_idx == 5:  # +x: i=ngll-1
        for j in range(ngll):
            for k in range(ngll):
                nodes.append(((ngll - 1) * ngll + j) * ngll + k)
    return nodes


def _face_gll_node_tuples(ngll: int) -> list[list[tuple[int, int, int]]]:
    """Return 6 face lists, each with (i,j,k) tuples for that face."""
    faces = []
    for f in range(6):
        nodes = _face_gll_nodes(f, ngll)
        tuples = []
        for n in nodes:
            k = n % ngll
            j = (n // ngll) % ngll
            i = n // (ngll * ngll)
            tuples.append((i, j, k))
        faces.append(tuples)
    return faces


# ── Topology-driven global node numbering ─────────────────────────────────


def compute_global_cell2global_node(
    topology: TopologyData, ngll: int
) -> tuple[npt.NDArray[np.int32], int]:
    """Assign global node IDs using topology face adjacency (no coordinates).

    Uses surface-sharing information from the mesh topology to match GLL
    nodes on shared faces between adjacent elements.  Assumes consistent
    hex element orientation (standard Gmsh output).  Face GLL nodes are
    matched by their position in the canonical face node lists — two
    adjacent elements' shared-face nodes correspond index-by-index.

    This replaces the old O(N log N) coordinate quantize-sort-dedup with
    an O(N_faces · ngll²) topology-driven pass.

    Args:
        topology:  Mesh topology with cell_to_surface.
        ngll:  Number of GLL nodes per dimension.

    Returns:
        global_cell2global_node: [n_cell, ngll, ngll, ngll] int32
        n_global_node: number of unique global nodes.
    """
    n_cell = topology.n_cell
    n_node = ngll * ngll * ngll
    c2s = topology.cell_to_surface

    global_ids = np.full((n_cell, ngll, ngll, ngll), -1, dtype=np.int32)

    # Precompute face GLL node lists as (i,j,k) tuples
    face_node_tuples = _face_gll_node_tuples(ngll)

    # Build surface → [(cell, face_idx)] and element → (surface→face) maps
    surf_to_cellface: dict[int, list[tuple[int, int]]] = {}
    elem_surf_to_face: list[dict[int, int]] = [{} for _ in range(n_cell)]

    for e in range(n_cell):
        for face_idx, signed_sid in enumerate(c2s[e]):
            abs_sid = abs(int(signed_sid)) - 1
            if abs_sid not in surf_to_cellface:
                surf_to_cellface[abs_sid] = []
            surf_to_cellface[abs_sid].append((e, face_idx))
            elem_surf_to_face[e][abs_sid] = face_idx

    next_id = 0

    # Pass 1: match shared-face nodes between adjacent elements
    for _, cellfaces in surf_to_cellface.items():
        if len(cellfaces) != 2:
            continue  # boundary surface — handled in pass 2

        (e0, f0), (e1, f1) = cellfaces[0], cellfaces[1]
        nodes0 = face_node_tuples[f0]
        nodes1 = face_node_tuples[f1]

        # Consistent orientation: nodes0[k] ↔ nodes1[k] positionally
        for (i0, j0, k0), (i1, j1, k1) in zip(nodes0, nodes1):
            gid0 = global_ids[e0, i0, j0, k0]
            gid1 = global_ids[e1, i1, j1, k1]

            if gid0 < 0 and gid1 < 0:
                global_ids[e0, i0, j0, k0] = next_id
                global_ids[e1, i1, j1, k1] = next_id
                next_id += 1
            elif gid0 >= 0 and gid1 < 0:
                global_ids[e1, i1, j1, k1] = gid0
            elif gid0 < 0 and gid1 >= 0:
                global_ids[e0, i0, j0, k0] = gid1
            # else: both already assigned (edge/corner shared by ≥2 faces)

    # Pass 2: assign remaining unassigned nodes (boundary faces + interior)
    unassigned = np.argwhere(global_ids < 0)  # [n_remain, 4] → (e,i,j,k)
    for idx in unassigned:
        e, i, j, k = int(idx[0]), int(idx[1]), int(idx[2]), int(idx[3])
        if global_ids[e, i, j, k] < 0:
            global_ids[e, i, j, k] = next_id
            next_id += 1

    n_global_node = next_id
    return global_ids, n_global_node


# ── Per-rank local node compaction (legacy — not used in pipeline) ────────


def compute_local_cell2rank_node(
    gll_coords: npt.NDArray[np.float64], element_ids: list[int]
) -> tuple[npt.NDArray[np.int32], int]:
    """Compute per-rank local_cell2rank_node mapping from GLL coordinates.

    Sorts GLL node coordinates for the specified elements, then assigns
    the same per-rank global node ID (0-based) to nodes at identical
    physical positions (SPECFEM get_global algorithm).  **Not used in the
    current pipeline** — kept for testing backward compatibility.  The
    pipeline now uses topology-based global numbering + per-rank slicing.

    Args:
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] — all elements' GLL coords.
        element_ids: 0-based element indices composing this rank
                     (local + ghost).  Local elements must come first.

    Returns:
        local_cell2rank_node: [n_elem, NGLL, NGLL, NGLL] int32
        n_rank_node: number of unique per-rank global nodes.
    """
    NGLL = gll_coords.shape[1]
    n_node = NGLL * NGLL * NGLL
    n_elem = len(element_ids)

    if n_elem == 0:
        return np.zeros((0, NGLL, NGLL, NGLL), dtype=np.int32), 0

    rank_coords = gll_coords[element_ids]

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
    coords_flat = rank_coords.reshape(n_points, 3)
    elem_idx = np.repeat(np.arange(n_elem, dtype=np.int32), n_node)
    node_idx = np.tile(np.arange(n_node, dtype=np.int32), n_elem)

    scale = np.float64(1.0) / tol
    iquan = np.rint(coords_flat * scale).astype(np.int64)
    order = np.lexsort((iquan[:, 2], iquan[:, 1], iquan[:, 0]))
    sorted_iquan = iquan[order]
    sorted_elem = elem_idx[order]
    sorted_node = node_idx[order]

    is_new = np.any(np.diff(sorted_iquan, axis=0, prepend=sorted_iquan[:1] - 1) != 0, axis=1)
    rank_node_id_vals = np.cumsum(is_new, dtype=np.int32) - 1

    local_cell2rank_node = np.zeros((n_elem, NGLL, NGLL, NGLL), dtype=np.int32)
    i_idx = sorted_node // (NGLL * NGLL)
    j_idx = (sorted_node // NGLL) % NGLL
    k_idx = sorted_node % NGLL
    local_cell2rank_node[sorted_elem, i_idx, j_idx, k_idx] = rank_node_id_vals

    n_rank_node = int(rank_node_id_vals[-1]) + 1 if n_points > 0 else 0
    return local_cell2rank_node, n_rank_node


# ── Main partition entry point ────────────────────────────────────────────


def compute_per_rank(
    topology: TopologyData,
    ngll: int,
    element_to_rank: npt.NDArray[np.int64],
    global_cell2global_node: npt.NDArray[np.int32],
) -> dict:
    """Compute per-rank data structures from an existing partition.

    Given element_to_rank and global_cell2global_node (from e.g. C++),
    computes local/ghost elements, exchange DOF patterns, and compact
    per-rank node numbering.  Does NOT call METIS — the partition must
    already exist.

    Returns the per_rank dict (same format as partition().per_rank).
    """
    n_cell = topology.n_cell
    n_surface = topology.n_surface
    c2s = topology.cell_to_surface
    n_ranks = int(element_to_rank.max()) + 1 if n_cell > 0 else 1

    # Init per_rank
    per_rank: dict[int, dict] = {}
    for rank in range(n_ranks):
        locals_list: list[int] = [e for e in range(n_cell) if element_to_rank[e] == rank]
        per_rank[rank] = {
            "local_cell_ids": locals_list,
            "ghost_cell_ids": [],
            "ghost_owners": [],
            "exchange": {},
        }

    # Surface → cells map
    surf_to_cells: dict[int, list[int]] = {}
    for e in range(n_cell):
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
                c1, c2 = cells_on_surf[i], cells_on_surf[j]
                r1, r2 = int(element_to_rank[c1]), int(element_to_rank[c2])
                if r1 == r2:
                    continue
                for owner_rank, ghost_cell in [(r1, c2), (r2, c1)]:
                    rd = per_rank[owner_rank]
                    if ghost_cell not in rd["ghost_cell_ids"]:
                        rd["ghost_cell_ids"].append(ghost_cell)
                        rd["ghost_owners"].append(int(element_to_rank[ghost_cell]))

    # ── Global node → owning ranks (via LOCAL cell ownership) ──
    # A GLL node shared by elements on multiple ranks must exchange its
    # residual/mass contributions with EVERY co-owning rank.  Face adjacency
    # alone misses edge/corner nodes shared by 3+ ranks (e.g. 8 ranks meeting
    # at one interior corner), so exchange patterns are built from global node
    # ownership in the compaction loop below.
    node_owner_ranks: dict[int, set[int]] = {}
    for cell_id in range(n_cell):
        owner_rank = int(element_to_rank[cell_id])
        for global_node in np.unique(global_cell2global_node[cell_id]):
            node_owner_ranks.setdefault(int(global_node), set()).add(owner_rank)

    # Per-rank compaction
    n_global_node = (
        int(global_cell2global_node.max()) + 1 if global_cell2global_node.size > 0 else 0
    )

    for rank in range(n_ranks):
        rd = per_rank[rank]
        locals_list = list(rd["local_cell_ids"])
        ghosts_list = list(rd["ghost_cell_ids"])

        all_elem_ids = locals_list + ghosts_list
        ibool_global_4d = global_cell2global_node[all_elem_ids]

        _, inverse = np.unique(ibool_global_4d.ravel(), return_inverse=True)
        ibool_compact_4d = inverse.reshape(ibool_global_4d.shape).astype(np.int32)
        n_rank_node = ibool_compact_4d.max() + 1 if ibool_compact_4d.size > 0 else 0

        rd["local_cell2rank_node"] = ibool_compact_4d
        rd["local_cell2global_node"] = ibool_global_4d
        rd["n_rank_node"] = n_rank_node

        # ── Exchange DOF patterns (node-ownership based, compact indices) ──
        # A GLL node shared by elements on multiple ranks must exchange its
        # residual/mass contributions with EVERY co-owning rank.  Face
        # adjacency alone misses edge/corner nodes shared by 3+ ranks, so
        # patterns are built from global node ownership.
        #
        # ORDERING CONTRACT: for each neighbor pair (rank, other_rank), both
        # sides must list shared nodes in the SAME order, because the solver
        # packs send buffers positionally.  Local compact indices differ
        # between ranks, so nodes are ordered by GLOBAL node id (identical
        # on both sides) before mapping to local compact DOFs.
        neighbor_shared_nodes: dict[int, dict[int, int]] = {}
        n_local = len(locals_list)
        for cell_row in range(n_local):
            for global_node, compact_node in zip(
                ibool_global_4d[cell_row].ravel(), ibool_compact_4d[cell_row].ravel()
            ):
                owners = node_owner_ranks[int(global_node)]
                if len(owners) < 2:
                    continue
                g = int(global_node)
                c = int(compact_node)
                for other_rank in owners:
                    if other_rank == rank:
                        continue
                    shared = neighbor_shared_nodes.setdefault(other_rank, {})
                    shared[g] = c  # dedup: same mapping for every shared cell

        exchange_dof: dict[int, dict] = {}
        for other_rank, shared in neighbor_shared_nodes.items():
            dofs: list[int] = []
            for g in sorted(shared):  # global-id order: matches the remote side
                base = shared[g] * 3
                dofs.extend((base, base + 1, base + 2))
            exchange_dof[other_rank] = {"send_dof": dofs, "recv_dof": list(dofs)}
        rd["exchange"] = exchange_dof

    return per_rank


def partition(topology: TopologyData, ngll: int, n_ranks: int) -> dict:
    """Partition elements across MPI ranks (METIS, topology-only).

    Builds a dual graph from cell adjacency, calls METIS k-way
    partitioning, then computes global node numbering from topology
    face adjacency.

    Args:
        topology:  Mesh topology.
        ngll:  Number of GLL nodes per dimension (polynomial_order + 1).
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
            local_cell2rank_node: [n_local+n_ghost, ngll, ngll, ngll] int32
            n_rank_node: int
            exchange: dict neighbor_rank → {
                "send_dof": list of per-rank global DOF indices (node_id*3+dir),
                "recv_dof": list of per-rank global DOF indices (node_id*3+dir),
            }
    """
    n_cell = topology.n_cell

    # ── METIS partition ──
    adjacency_list, _ = _build_dual_graph(topology)

    if n_ranks > 1 and n_cell > 1:
        part_result = pymetis.part_graph(n_ranks, adjacency=adjacency_list, recursive=True)
        element_to_rank = np.array(part_result[1], dtype=np.int64)
    else:
        element_to_rank = np.zeros(n_cell, dtype=np.int64)

    # ── Global node numbering ──
    global_cell2global_node, _n_global = compute_global_cell2global_node(topology, ngll)

    # ── Per-rank computation ──
    per_rank = compute_per_rank(topology, ngll, element_to_rank, global_cell2global_node)

    return {
        "element_to_rank": element_to_rank,
        "n_ranks": n_ranks,
        "per_rank": per_rank,
        "global_cell2global_node": global_cell2global_node,
        "n_global_node": _n_global,
    }
