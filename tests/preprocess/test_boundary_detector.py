"""Tests for boundary_detector module."""

import os
import sys

import numpy as np

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.boundary_detector import detect_boundaries


def _make_unit_cube_topo():
    """Create TopologyData for a unit cube [0,1]^3 with proper surface definitions."""
    from preprocess.topology_reader import TopologyData

    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=np.float64,
    )

    e2v = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],  # 1-4: bottom (z=0)
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 5],  # 5-8: top (z=1)
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8],  # 9-12: vertical
        ],
        dtype=np.int64,
    )

    # Each face formed by a closed loop of 4 edges
    s2e = np.array(
        [
            [1, 2, 3, 4],  # -z (z=0): {1,2,3,4}
            [5, 6, 7, 8],  # +z (z=1): {5,6,7,8}
            [1, 10, -5, -9],  # -y (y=0): {1,2,6,5}
            [3, 12, -7, -11],  # +y (y=1): {3,4,8,7}
            [-4, 12, -8, -9],  # -x (x=0): {1,4,8,5}
            [2, 11, -6, -10],  # +x (x=1): {2,3,7,6}
        ],
        dtype=np.int64,
    )

    c2s = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    return TopologyData(verts, e2v, s2e, c2s, 8, 12, 6, 1)


def _make_two_cube_topo():
    """Two cubes stacked in z: [0,1]^2 x [0,2], splitting at z=1."""
    from preprocess.topology_reader import TopologyData

    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
        ],
        dtype=np.float64,
    )

    e2v = np.array(
        [
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],  # 1-4
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 5],  # 5-8
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8],  # 9-12
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 9],  # 13-16
            [5, 9],
            [6, 10],
            [7, 11],
            [8, 12],  # 17-20
        ],
        dtype=np.int64,
    )

    # Bottom cell (1-8): surfaces 1-6  Top cell (5-12): surfaces 7-11 (shared face 2 reused)
    s2e = np.array(
        [
            [1, 2, 3, 4],  #  1: -z (z=0): {1,2,3,4}
            [5, 6, 7, 8],  #  2: +z shared (z=1): {5,6,7,8}
            [1, 10, -5, -9],  #  3: -y: {1,2,6,5}
            [3, 12, -7, -11],  #  4: +y: {3,4,8,7}
            [-4, 12, -8, -9],  #  5: -x: {1,4,8,5}
            [2, 11, -6, -10],  #  6: +x: {2,3,7,6}
            [13, 14, 15, 16],  #  7: +z (z=2): {9,10,11,12}
            [5, 18, -13, -17],  #  8: -y (top): {5,6,10,9}
            [7, 20, -15, -19],  #  9: +y (top): {7,8,12,11}
            [-8, 20, 16, -17],  # 10: -x (top): {5,8,12,9}
            [6, 19, -14, -18],  # 11: +x (top): {6,7,11,10}
        ],
        dtype=np.int64,
    )

    c2s = np.array([[1, 2, 3, 4, 5, 6], [-2, 7, 8, 9, 10, 11]], dtype=np.int64)

    return TopologyData(verts, e2v, s2e, c2s, 12, 20, 11, 2)


class TestBoundaryDetector:
    def test_single_cube_all_boundaries(self):
        """Unit cube at [0,1]^3 -- zmin=0 free surface, others absorbing."""
        topo = _make_unit_cube_topo()
        bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        boundary_tag, is_pml = detect_boundaries(topo, bounds)

        assert boundary_tag.shape == (6,)
        assert np.sum(boundary_tag == 1) == 1  # one free surface at zmin
        assert np.sum(boundary_tag == 2) == 5  # other 5 absorbing
        assert np.sum(boundary_tag == 0) == 0  # no interior

        assert is_pml.shape == (1,)
        assert is_pml[0] == True

    def test_free_surface_at_zmin(self):
        """Verify free surface is at zmin."""
        topo = _make_unit_cube_topo()
        bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}
        boundary_tag, _ = detect_boundaries(topo, bounds)

        free_idx = np.where(boundary_tag == 1)[0]
        assert len(free_idx) == 1

        s2e = topo.surface_to_edge
        e2v = topo.edge_to_vertex
        v2c = topo.vertex_to_coord
        vids = set()
        for sid in s2e[free_idx[0]]:
            eid = abs(int(sid)) - 1
            vids.add(int(e2v[eid, 0]))
            vids.add(int(e2v[eid, 1]))
        face_coords = np.array([v2c[v - 1] for v in sorted(vids)])
        assert np.isclose(face_coords[:, 2].mean(), 0.0, atol=1e-6)

    def test_two_cubes_shared_face_interior(self):
        """Two cubes stacked: shared face at z=1 is interior (tag 0)."""
        topo = _make_two_cube_topo()
        bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}
        boundary_tag, is_pml = detect_boundaries(topo, bounds)

        assert boundary_tag.shape == (11,)
        assert np.sum(boundary_tag == 1) == 1  # free surface at zmin
        assert np.sum(boundary_tag == 0) == 1  # shared face interior
        assert np.sum(boundary_tag == 2) == 9  # all other faces

        assert is_pml.shape == (2,)
        assert np.all(is_pml)
