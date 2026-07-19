"""Tests for GLL-node recording map generation."""

import numpy as np
from preprocess.recording_map import build_recording_map
from preprocess.topology_reader import TopologyData


def test_build_recording_map_selects_gll_nodes():
    """Recording map should select all NGLL^3 GLL nodes per recording cell."""
    ngll = 2
    n_cell = 2

    # global_cell2global_node: [n_cell, ngll, ngll, ngll]
    # cell 0 has nodes 0-7, cell 1 has nodes 4-11 (4 shared on the interface)
    global_cell2global_node = np.zeros((n_cell, ngll, ngll, ngll), dtype=np.int64)
    global_cell2global_node[0] = np.arange(8).reshape(ngll, ngll, ngll)
    global_cell2global_node[1] = np.arange(4, 12).reshape(ngll, ngll, ngll)

    # coords: [n_cell, ngll, ngll, ngll, 3] — node positions in 3D
    coords = np.zeros((n_cell, ngll, ngll, ngll, 3), dtype=np.float64)
    for cell in range(n_cell):
        for i in range(ngll):
            for j in range(ngll):
                for k in range(ngll):
                    node = global_cell2global_node[cell, i, j, k]
                    coords[cell, i, j, k] = [float(node % 4), float((node // 4) % 4), 0.0]

    # Minimal topology: 2 cells
    topology = TopologyData(
        vertex_to_coord=np.zeros((12, 3), dtype=np.float64),
        edge_to_vertex=np.array([[0, 1]], dtype=np.int64),
        surface_to_edge=np.zeros((12, 4), dtype=np.int64),
        cell_to_surface=np.zeros((2, 6), dtype=np.int64),
        n_vertex=12,
        n_edge=1,
        n_surface=12,
        n_cell=n_cell,
    )

    # Both cells in recording region (depth = full domain, z=0)
    is_pml = np.zeros(n_cell, dtype=np.int8)
    domain_bounds = {
        "zmin": 0.0,
        "zmax": 100.0,
        "xmin": 0.0,
        "xmax": 200.0,
        "ymin": 0.0,
        "ymax": 100.0,
    }

    result = build_recording_map(
        topology=topology,
        domain_bounds=domain_bounds,
        is_pml=is_pml,
        record_depth_max_m=100.0,
        global_cell2global_node=global_cell2global_node,
        gll_node_coords=coords,
    )

    rec = result["per_rank_recording"][0]
    # Both cells recorded
    assert len(rec["rec_cell_global_ids"]) == 2
    assert len(rec["rec_cell_local_index"]) == 2
    # cell_gll_node_ids: [2, 8]
    assert len(rec["cell_gll_node_ids"]) == 2
    assert len(rec["cell_gll_node_ids"][0]) == 8
    # Unique GLL nodes: 0-11 = 12 unique
    assert len(rec["gll_node_ids"]) == 12
    # cell_gll_node_index: [2, 8], values index into gll_node_ids
    assert len(rec["cell_gll_node_index"]) == 2
    for cell_index in range(2):
        for node_index in range(8):
            idx = rec["cell_gll_node_index"][cell_index][node_index]
            assert 0 <= idx < 12
            # Index maps to correct global node ID
            expected_node = int(global_cell2global_node[cell_index].ravel()[node_index])
            assert rec["gll_node_ids"][idx] == expected_node
    # gll_node_coords: [12, 3]
    assert len(rec["gll_node_coords"]) == 12
    assert len(rec["gll_node_coords"][0]) == 3


def test_single_cell_gll_nodes():
    """Single cell with NGLL=3 should select 27 unique GLL nodes."""
    ngll = 3
    n_cell = 1
    n_node = ngll**3  # 27

    global_cell2global_node = np.arange(n_node, dtype=np.int64).reshape(n_cell, ngll, ngll, ngll)

    coords = np.zeros((n_cell, ngll, ngll, ngll, 3), dtype=np.float64)
    for cell in range(n_cell):
        for i in range(ngll):
            for j in range(ngll):
                for k in range(ngll):
                    node = global_cell2global_node[cell, i, j, k]
                    coords[cell, i, j, k] = [
                        float(node % ngll),
                        float((node // ngll) % ngll),
                        float(node // (ngll * ngll)),
                    ]

    topology = TopologyData(
        vertex_to_coord=np.array([[float(i), 0.0, 0.0] for i in range(n_node)], dtype=np.float64),
        edge_to_vertex=np.array([[0, 1]], dtype=np.int64),
        surface_to_edge=np.zeros((6, 4), dtype=np.int64),
        cell_to_surface=np.zeros((1, 6), dtype=np.int64),
        n_vertex=n_node,
        n_edge=1,
        n_surface=6,
        n_cell=n_cell,
    )

    is_pml = np.zeros(n_cell, dtype=np.int8)
    domain_bounds = {
        "zmin": 0.0,
        "zmax": 10.0,
        "xmin": 0.0,
        "xmax": 10.0,
        "ymin": 0.0,
        "ymax": 10.0,
    }

    result = build_recording_map(
        topology=topology,
        domain_bounds=domain_bounds,
        is_pml=is_pml,
        record_depth_max_m=10.0,
        global_cell2global_node=global_cell2global_node,
        gll_node_coords=coords,
    )

    rec = result["per_rank_recording"][0]
    assert len(rec["rec_cell_global_ids"]) == 1
    assert len(rec["gll_node_ids"]) == 27  # all unique
    assert len(rec["cell_gll_node_index"]) == 1
    assert len(rec["cell_gll_node_index"][0]) == 27


def test_pml_exclusion():
    """PML cell excluded from recording."""
    ngll = 2
    n_cell = 1

    global_cell2global_node = np.arange(8, dtype=np.int64).reshape(n_cell, ngll, ngll, ngll)
    coords = np.zeros((n_cell, ngll, ngll, ngll, 3), dtype=np.float64)
    for i in range(ngll):
        for j in range(ngll):
            for k in range(ngll):
                coords[0, i, j, k] = [float(i), float(j), float(k)]

    topology = TopologyData(
        vertex_to_coord=np.zeros((8, 3), dtype=np.float64),
        edge_to_vertex=np.array([[0, 1]], dtype=np.int64),
        surface_to_edge=np.zeros((6, 4), dtype=np.int64),
        cell_to_surface=np.zeros((1, 6), dtype=np.int64),
        n_vertex=8,
        n_edge=1,
        n_surface=6,
        n_cell=n_cell,
    )

    domain_bounds = {"zmin": 0.0, "zmax": 1.0, "xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
    is_pml = np.array([True], dtype=bool)

    result = build_recording_map(
        topology=topology,
        domain_bounds=domain_bounds,
        is_pml=is_pml,
        record_depth_max_m=1.0,
        global_cell2global_node=global_cell2global_node,
        gll_node_coords=coords,
    )

    rec = result["per_rank_recording"][0]
    assert len(rec["rec_cell_global_ids"]) == 0
    assert len(rec["gll_node_ids"]) == 0
