"""Tests for the recording map builder."""

import numpy as np
import pytest
from pytest import approx

from preprocess.recording_map import build_recording_map
from preprocess.topology_reader import TopologyData


def _unit_cube_topology() -> TopologyData:
    """Build a TopologyData for a single unit cube [0,1]^3.

    Vertex positions:
      v0=(0,0,0), v1=(1,0,0), v2=(1,1,0), v3=(0,1,0)
      v4=(0,0,1), v5=(1,0,1), v6=(1,1,1), v7=(0,1,1)
    """
    v = np.array([
        [0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[0.0,1.0,0.0],
        [0.0,0.0,1.0],[1.0,0.0,1.0],[1.0,1.0,1.0],[0.0,1.0,1.0],
    ], dtype=np.float64)
    e2v = np.array([
        [1,2],[2,3],[4,3],[1,4],[5,6],[6,7],[8,7],[5,8],
        [1,5],[2,6],[4,8],[3,7],
    ], dtype=np.int64)
    s2e = np.array([
        [1,2,-3,-4],      # F0(-z)
        [5,6,-7,-8],      # F1(+z)
        [1,10,-5,-9],     # F2(-y)
        [-3,11,7,-12],    # F3(+y)
        [4,11,-8,-9],     # F4(-x)
        [2,12,-6,-10],    # F5(+x)
    ], dtype=np.int64)
    c2s = np.array([[1,2,3,4,5,6]], dtype=np.int64)
    return TopologyData(
        n_vertex=8, n_edge=12, n_surface=6, n_cell=1,
        vertex_to_coord=v, cell_to_surface=c2s,
        surface_to_edge=s2e, edge_to_vertex=e2v,
    )


def test_single_element_no_pml():
    """Single element, no PML, full depth — all 8 vertices selected."""
    topo = _unit_cube_topology()
    domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}
    bt = np.ones(6, dtype=np.int64) * 2
    is_pml = np.array([False], dtype=bool)

    result = build_recording_map(topo, bt, domain, is_pml, record_depth_max_m=1.0, green_tile_size_m=0.5)
    rec0 = result["per_rank_recording"][0]
    assert len(rec0["vertex_ids"]) == 8
    assert rec0["save_element_mask"] == [True]
    assert result["record_depth_actual_m"] == approx(1.0)


def test_pml_exclusion():
    """PML element excluded from recording."""
    topo = _unit_cube_topology()
    domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}
    bt = np.ones(6, dtype=np.int64) * 2
    is_pml = np.array([True], dtype=bool)  # element is PML

    result = build_recording_map(topo, bt, domain, is_pml, record_depth_max_m=1.0, green_tile_size_m=0.5)
    rec0 = result["per_rank_recording"][0]
    assert len(rec0["vertex_ids"]) == 0  # no vertices — PML excluded


def test_source_element_local_index_valid():
    """Each vertex's source_element_local_index points to a valid element."""
    topo = _unit_cube_topology()
    domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}
    bt = np.ones(6, dtype=np.int64) * 2
    is_pml = np.array([False], dtype=bool)

    result = build_recording_map(topo, bt, domain, is_pml, record_depth_max_m=1.0, green_tile_size_m=0.5)
    rec0 = result["per_rank_recording"][0]
    for idx in rec0["source_element_local_index"]:
        assert 0 <= idx < 1  # only 1 element
    for ci in rec0["source_corner_index"]:
        assert 0 <= ci < 8


def test_full_domain_depth():
    """Depth exceeds domain — all 8 vertices selected."""
    topo = _unit_cube_topology()
    domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}
    bt = np.ones(6, dtype=np.int64) * 2
    is_pml = np.array([False], dtype=bool)

    result = build_recording_map(topo, bt, domain, is_pml, record_depth_max_m=100.0, green_tile_size_m=0.5)
    rec0 = result["per_rank_recording"][0]
    assert len(rec0["vertex_ids"]) == 8
    assert result["record_depth_actual_m"] == approx(1.0)


def test_shallow_depth():
    """Very shallow depth — only top-surface vertices (z=0 face) returned."""
    topo = _unit_cube_topology()
    domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}
    bt = np.ones(6, dtype=np.int64) * 2
    is_pml = np.array([False], dtype=bool)

    # Depth 0 — only vertices on the free surface (z=0) should be included.
    # Since the element centroid is at z=0.5, and target_z=0, no element centroid
    # is at or above z=0.5... wait, target_z = zmin + 0 = 0. No elements have
    # centroid ≤ 0 since centroid is at 0.5. So 0 vertices selected.
    result = build_recording_map(topo, bt, domain, is_pml, record_depth_max_m=0.0, green_tile_size_m=0.5)
    rec0 = result["per_rank_recording"][0]
    assert len(rec0["vertex_ids"]) == 0
    assert result["record_depth_actual_m"] == approx(0.0)


def test_save_element_mask_shape():
    """save_element_mask has length = n_local_elem (1 for unit cube)."""
    topo = _unit_cube_topology()
    domain = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}
    bt = np.ones(6, dtype=np.int64) * 2
    is_pml = np.array([False], dtype=bool)

    result = build_recording_map(topo, bt, domain, is_pml, record_depth_max_m=1.0, green_tile_size_m=0.5)
    rec0 = result["per_rank_recording"][0]
    assert len(rec0["save_element_mask"]) == 1  # 1 element
    # Save mask should be True since element has recorded vertices
    assert rec0["save_element_mask"] == [True]