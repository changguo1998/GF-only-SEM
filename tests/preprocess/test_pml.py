"""Tests for pml damping module."""

import os
import sys

import numpy as np
import pytest

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.pml import compute_pml_damping


def _make_unit_cube_topo():
    """Create TopologyData for a unit cube [0,1]^3."""
    from preprocess.topology_reader import TopologyData
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)

    e2v = np.array([
        [1, 2], [2, 3], [3, 4], [4, 1],
        [5, 6], [6, 7], [7, 8], [8, 5],
        [1, 5], [2, 6], [3, 7], [4, 8],
    ], dtype=np.int64)

    s2e = np.array([
        [1, 2, 3, 4],          # 1: -z (z=0)
        [5, 6, 7, 8],          # 2: +z (z=1)
        [1, 10, -5, -9],       # 3: -y (y=0)
        [3, 12, -7, -11],      # 4: +y (y=1)
        [-4, 12, -8, -9],      # 5: -x (x=0)
        [2, 11, -6, -10],      # 6: +x (x=1)
    ], dtype=np.int64)

    c2s = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
    return TopologyData(verts, e2v, s2e, c2s, 8, 12, 6, 1)


def _make_two_cube_topo():
    """Two cubes stacked in z: [0,1]^2 x [0,2], splitting at z=1.
    Same topology as test_boundary_detector for consistency.
    """
    from preprocess.topology_reader import TopologyData
    verts = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2],
    ], dtype=np.float64)

    e2v = np.array([
        [1, 2], [2, 3], [3, 4], [4, 1],
        [5, 6], [6, 7], [7, 8], [8, 5],
        [1, 5], [2, 6], [3, 7], [4, 8],
        [9, 10], [10, 11], [11, 12], [12, 9],
        [5, 9], [6, 10], [7, 11], [8, 12],
    ], dtype=np.int64)

    s2e = np.array([
        [1, 2, 3, 4],            #  1: -z (z=0)
        [5, 6, 7, 8],            #  2: +z shared (z=1)
        [1, 10, -5, -9],         #  3: -y
        [3, 12, -7, -11],        #  4: +y
        [-4, 12, -8, -9],        #  5: -x
        [2, 11, -6, -10],        #  6: +x
        [13, 14, 15, 16],        #  7: +z (z=2)
        [5, 18, -13, -17],       #  8: -y (top)
        [7, 20, -15, -19],       #  9: +y (top)
        [-8, 20, 16, -17],       # 10: -x (top)
        [6, 19, -14, -18],       # 11: +x (top)
    ], dtype=np.int64)

    c2s = np.array([
        [1, 2, 3, 4, 5, 6],
        [-2, 7, 8, 9, 10, 11],
    ], dtype=np.int64)

    return TopologyData(verts, e2v, s2e, c2s, 12, 20, 11, 2)


class TestPMLDamping:
    """Test compute_pml_damping function."""

    def test_non_pml_cell_returns_zero(self):
        """Non-PML cells should have zero damping everywhere."""
        topo = _make_unit_cube_topo()
        from preprocess.gll_geometry import compute_gll_geometry
        coords, _, _, _ = compute_gll_geometry(topo, N=2)  # NGLL=3

        is_pml = np.array([False], dtype=bool)
        pml_thickness = {"xmin": 1, "xmax": 1, "ymin": 1, "ymax": 1, "zmin": 0, "zmax": 1}
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

        damping = compute_pml_damping(topo, coords, pml_thickness, domain_bounds, is_pml)
        assert damping.shape == (1, 3, 3, 3)
        assert np.all(damping == 0.0)

    def test_pml_cell_has_nonzero_damping_near_boundary(self):
        """PML cell should have nonzero damping near absorbing boundary."""
        topo = _make_two_cube_topo()
        from preprocess.gll_geometry import compute_gll_geometry
        coords, _, _, _ = compute_gll_geometry(topo, N=2)  # NGLL=3

        # Cell 0: z in [0,1], Cell 1: z in [1,2]
        # PML on zmax (z=2) with thickness 1
        is_pml = np.array([False, True], dtype=bool)
        pml_thickness = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 1}
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}

        damping = compute_pml_damping(topo, coords, pml_thickness, domain_bounds, is_pml)

        assert damping.shape == (2, 3, 3, 3)

        # Cell 0 is not PML → all zeros
        assert np.all(damping[0] == 0.0)

        # Cell 1 is PML (z in [1,2]): highest near z=2 (k=2), zero near z=1 (k=0)
        damp_cell1 = damping[1]
        assert np.all(damp_cell1[:, :, 2] > 0), "PML boundary nodes should have damping > 0"
        assert np.all(damp_cell1[:, :, 0] == 0), "PML entry nodes should have zero damping"

    def test_pml_damping_monotonic_ramp(self):
        """Damping should increase monotonically from PML entry to boundary."""
        topo = _make_two_cube_topo()
        from preprocess.gll_geometry import compute_gll_geometry
        coords, _, _, _ = compute_gll_geometry(topo, N=3)  # NGLL=4

        is_pml = np.array([False, True], dtype=bool)
        pml_thickness = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 1}
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}

        damping = compute_pml_damping(topo, coords, pml_thickness, domain_bounds, is_pml)

        # Cell 1: damping along z should increase from z=1 to z=2
        damp_cell1 = damping[1, 2, 2, :]  # along z for fixed xi, eta
        assert damp_cell1[0] == 0.0  # PML entry
        for i in range(len(damp_cell1) - 1):
            assert damp_cell1[i + 1] >= damp_cell1[i], \
                f"Damping not monotonic: {damp_cell1}"
        assert damp_cell1[-1] > 0.0  # PML boundary

    def test_mixed_pml_and_non_pml_elements(self):
        """Only PML elements should have nonzero damping."""
        topo = _make_two_cube_topo()
        from preprocess.gll_geometry import compute_gll_geometry
        coords, _, _, _ = compute_gll_geometry(topo, N=2)

        is_pml = np.array([False, True], dtype=bool)
        pml_thickness = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 1}
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}

        damping = compute_pml_damping(topo, coords, pml_thickness, domain_bounds, is_pml)

        assert np.all(damping[0] == 0.0)  # non-PML
        assert np.any(damping[1] > 0)    # PML

    def test_no_pml_thickness_returns_all_zero(self):
        """When pml_thickness is all zero, all damping should be zero."""
        topo = _make_two_cube_topo()
        from preprocess.gll_geometry import compute_gll_geometry
        coords, _, _, _ = compute_gll_geometry(topo, N=2)

        is_pml = np.array([True, True], dtype=bool)
        pml_thickness = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}

        damping = compute_pml_damping(topo, coords, pml_thickness, domain_bounds, is_pml)
        assert np.all(damping == 0.0)

    def test_all_faces_pml_thickness(self):
        """PML on multiple faces should accumulate correctly."""
        topo = _make_unit_cube_topo()
        from preprocess.gll_geometry import compute_gll_geometry
        coords, _, _, _ = compute_gll_geometry(topo, N=2)

        is_pml = np.array([True], dtype=bool)
        pml_thickness = {"xmin": 1, "xmax": 1, "ymin": 1, "ymax": 1, "zmin": 0, "zmax": 1}
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

        damping = compute_pml_damping(topo, coords, pml_thickness, domain_bounds, is_pml)

        # Single element with all faces as PML → damping should be nonzero
        assert np.any(damping[0] > 0)
        # Max damping should be 1.0 (at corners touching multiple PML faces)
        assert np.isclose(damping[0].max(), 1.0, atol=1e-6)