"""Tests for source locator — surface and buried mode."""

import numpy as np
import pytest

from preprocess.gll_geometry import gll_quadrature_points
from preprocess.source_locator import (
    _find_candidate_elements,
    _get_element_corners,
    _lagrange_basis_at,
    _newton_find_xi,
    compute_source_weights,
    locate_source,
)
from preprocess.topology_reader import TopologyData

# ---------------------------------------------------------------------------
# Helpers: build small meshes for testing
# ---------------------------------------------------------------------------


def _two_cell_topology_and_coords(N=3):
    """Two hex elements side-by-side in x: [0,1]^3 and [1,2]^3.

    Returns (topology, gll_coords, boundary_tag).
    """
    NGLL = N + 1
    n_cell = 2
    n_surface = 12

    c2s = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.int64)
    bt = np.full(n_surface, 2, dtype=np.int64)
    bt[4] = 1  # cell0 face 5 (z=0)
    bt[10] = 1  # cell1 face 5 (z=0)

    e2v = np.zeros((12, 2), dtype=np.int64)
    s2e = np.zeros((n_surface, 4), dtype=np.int64)

    topology = TopologyData(
        n_vertex=12,
        n_edge=12,
        n_surface=n_surface,
        n_cell=n_cell,
        vertex_to_coord=np.zeros((12, 3), dtype=np.float64),
        cell_to_surface=c2s,
        surface_to_edge=s2e,
        edge_to_vertex=e2v,
    )

    # GLL coordinates using actual GLL points (include endpoints -1, +1)
    gll_1d = gll_quadrature_points(N)
    gll_coords = np.zeros((n_cell, NGLL, NGLL, NGLL, 3))
    for cell_idx, x0 in enumerate([0.0, 1.0]):
        for i in range(NGLL):
            for j in range(NGLL):
                for k in range(NGLL):
                    gll_coords[cell_idx, i, j, k, 0] = 0.5 * (gll_1d[i] + 1.0) + x0
                    gll_coords[cell_idx, i, j, k, 1] = 0.5 * (gll_1d[j] + 1.0)
                    gll_coords[cell_idx, i, j, k, 2] = 0.5 * (gll_1d[k] + 1.0)

    return topology, gll_coords, bt


def _four_cell_surface_topology_and_coords(N=3):
    """4 elements in a 2x2x1 slab sharing central vertex (1,1,0).

    Returns (topology, gll_coords, boundary_tag).
    """
    NGLL = N + 1
    n_cell = 4
    n_surface = 24

    c2s = np.arange(1, n_surface + 1, dtype=np.int64).reshape(n_cell, 6)
    bt = np.ones(n_surface, dtype=np.int64)  # all free surface

    e2v = np.zeros((12, 2), dtype=np.int64)
    s2e = np.zeros((n_surface, 4), dtype=np.int64)

    topology = TopologyData(
        n_vertex=18,
        n_edge=12,
        n_surface=n_surface,
        n_cell=n_cell,
        vertex_to_coord=np.zeros((18, 3), dtype=np.float64),
        cell_to_surface=c2s,
        surface_to_edge=s2e,
        edge_to_vertex=e2v,
    )

    gll_1d = gll_quadrature_points(N)
    gll_coords = np.zeros((n_cell, NGLL, NGLL, NGLL, 3))
    origins = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for cell_idx, (x0, y0) in enumerate(origins):
        for i in range(NGLL):
            for j in range(NGLL):
                for k in range(NGLL):
                    gll_coords[cell_idx, i, j, k, 0] = 0.5 * (gll_1d[i] + 1.0) + x0
                    gll_coords[cell_idx, i, j, k, 1] = 0.5 * (gll_1d[j] + 1.0) + y0
                    gll_coords[cell_idx, i, j, k, 2] = 0.5 * (gll_1d[k] + 1.0)

    return topology, gll_coords, bt


# ---------------------------------------------------------------------------
# Tests for low-level helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_lagrange_basis_cardinal(self):
        """Lagrange basis at GLL node i equals 1, at others equals 0."""
        gll_pts = gll_quadrature_points(3)
        for i, xi_i in enumerate(gll_pts):
            w = _lagrange_basis_at(xi_i, gll_pts)
            assert abs(w[i] - 1.0) < 1e-14
            for j in range(len(gll_pts)):
                if j != i:
                    assert abs(w[j]) < 1e-14

    def test_compute_source_weights_sum(self):
        """Lagrange weights sum to ~1 for any interior point."""
        gll_pts = gll_quadrature_points(4)
        for xi in [-0.5, 0.0, 0.3, 0.7]:
            for eta in [-0.5, 0.0, 0.3]:
                for zeta in [-0.5, 0.0, 0.3]:
                    w = compute_source_weights(np.array([xi, eta, zeta]), gll_pts)
                    assert abs(np.sum(w) - 1.0) < 1e-12

    def test_newton_find_xi(self):
        """Newton finds correct natural coords for interior point."""
        corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [2.0, 0.0, 2.0],
                [2.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
            ]
        )
        xi = _newton_find_xi(np.array([1.0, 1.0, 1.0]), corners)
        assert xi is not None
        assert np.allclose(xi, [0.0, 0.0, 0.0], atol=1e-10)

    def test_newton_rejects_outside(self):
        """Newton returns None for point outside element."""
        corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        assert _newton_find_xi(np.array([10.0, 10.0, 10.0]), corners) is None

    def test_get_element_corners(self):
        """_get_element_corners extracts GMSH-ordered corners."""
        _, coords, _ = _two_cell_topology_and_coords(3)
        corners = _get_element_corners(coords, 0, 4)
        assert corners.shape == (8, 3)
        # With GLL points, first corner is at x=0, y=0, z=0
        assert np.allclose(corners[0], [0.0, 0.0, 0.0])
        assert np.allclose(corners[6], [1.0, 1.0, 1.0])

    def test_find_candidate_elements_interior(self):
        """AABB search finds the correct element for interior point."""
        _, coords, _ = _two_cell_topology_and_coords(3)
        is_pml = np.zeros(2, dtype=bool)
        cand = _find_candidate_elements(np.array([0.5, 0.5, 0.5]), coords, is_pml)
        assert cand == [0]
        cand = _find_candidate_elements(np.array([1.5, 0.5, 0.5]), coords, is_pml)
        assert cand == [1]

    def test_find_candidate_elements_excludes_pml(self):
        """AABB search excludes PML elements."""
        _, coords, _ = _two_cell_topology_and_coords(3)
        is_pml = np.array([True, False], dtype=bool)
        cand = _find_candidate_elements(np.array([0.5, 0.5, 0.5]), coords, is_pml)
        assert cand == []


# ---------------------------------------------------------------------------
# Tests for locate_source
# ---------------------------------------------------------------------------


class TestLocateSourceSurface:
    """Surface mode (default, no is_pml or z matches zmin)."""

    def test_surface_source_at_vertex_four_elements(self):
        """Source at (1,1,0) shared by 4 surface elements."""
        topology, gll_coords, bt = _four_cell_surface_topology_and_coords(3)
        result = locate_source(topology, np.array([1.0, 1.0, 0.0]), gll_coords, bt, N=3)
        assert result["n_src_elem"] == 4
        assert result["mode"] == "surface"
        total = float(np.sum(result["weights"]))
        assert abs(total - 1.0) < 1e-10

    def test_surface_source_legacy_no_is_pml(self):
        """locate_source without is_pml argument."""
        topology, gll_coords, bt = _four_cell_surface_topology_and_coords(3)
        result = locate_source(topology, np.array([0.5, 0.5, 0.0]), gll_coords, bt, N=3)
        assert result["n_src_elem"] >= 1
        assert result["mode"] == "surface"


class TestLocateSourceBuried:
    """Buried mode (is_pml provided, z != zmin)."""

    @pytest.fixture
    def two_cell_setup(self):
        topology, gll_coords, bt = _two_cell_topology_and_coords(3)
        is_pml = np.zeros(2, dtype=bool)
        return topology, gll_coords, bt, is_pml

    def test_buried_source_single_element(self, two_cell_setup):
        """Buried source at element center."""
        topology, gll_coords, bt, is_pml = two_cell_setup
        result = locate_source(
            topology, np.array([0.5, 0.5, 0.5]), gll_coords, bt, N=3, is_pml=is_pml
        )
        assert result["n_src_elem"] == 1
        assert result["mode"] == "buried"
        assert result["element_ids"][0] == 1
        wsum = float(np.sum(result["weights"][0]))
        assert abs(wsum - 1.0) < 1e-10

    def test_buried_source_second_element(self, two_cell_setup):
        """Buried source in element 1."""
        topology, gll_coords, bt, is_pml = two_cell_setup
        result = locate_source(
            topology, np.array([1.5, 0.5, 0.5]), gll_coords, bt, N=3, is_pml=is_pml
        )
        assert result["n_src_elem"] == 1
        assert result["element_ids"][0] == 2

    def test_buried_source_in_pml_raises(self, two_cell_setup):
        """Buried source in PML element raises ValueError."""
        topology, gll_coords, bt, is_pml = two_cell_setup
        is_pml[0] = True
        with pytest.raises(ValueError, match="not contained in any non-PML"):
            locate_source(topology, np.array([0.5, 0.5, 0.5]), gll_coords, bt, N=3, is_pml=is_pml)

    def test_buried_source_outside_domain_raises(self, two_cell_setup):
        """Source outside domain raises ValueError."""
        topology, gll_coords, bt, is_pml = two_cell_setup
        with pytest.raises(ValueError, match="non-PML"):
            locate_source(
                topology, np.array([10.0, 10.0, 10.0]), gll_coords, bt, N=3, is_pml=is_pml
            )

    def test_buried_source_without_is_pml_uses_surface_mode(self, two_cell_setup):
        """With is_pml=None, buried z falls back to surface mode."""
        topology, gll_coords, bt, is_pml = two_cell_setup
        # z=0.5 matches zmin=0.0 → not considered buried. is_pml=None → surface mode.
        # Surface mode finds cell0 (it has a free-surface face) and cell0 contains (0.5,0.5,0.5).
        result = locate_source(
            topology, np.array([0.5, 0.5, 0.5]), gll_coords, bt, N=3, is_pml=None
        )
        assert result["n_src_elem"] == 1
        assert result["mode"] == "surface"
