"""Tests for gf_post.search — point-in-hexahedron Newton iteration."""

import numpy as np
import pytest
from gf_post.search import find_containing_element


class TestFindContainingElement:
    def test_centroid(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            coords = gr.coords
            dxi_dx = gr.dxi_dx
        point = np.array([0.5, 0.5, 0.5])
        candidates = np.array([1])  # 1-based
        eid, xi, eta, zeta = find_containing_element(point, candidates, coords, dxi_dx)
        assert eid == 1
        assert np.isclose(xi, 0.0, atol=1e-6)
        assert np.isclose(eta, 0.0, atol=1e-6)
        assert np.isclose(zeta, 0.0, atol=1e-6)

    def test_corner(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            coords = gr.coords
            dxi_dx = gr.dxi_dx
        point = np.array([0.0, 0.0, 0.0])
        candidates = np.array([1])
        eid, xi, eta, zeta = find_containing_element(point, candidates, coords, dxi_dx)
        assert eid == 1
        assert np.isclose(xi, -1.0, atol=1e-6)
        assert np.isclose(eta, -1.0, atol=1e-6)
        assert np.isclose(zeta, -1.0, atol=1e-6)

    def test_arbitrary_point(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            coords = gr.coords
            dxi_dx = gr.dxi_dx
        # Point at (0.25, 0.75, 0.5) → natural (ξ=-0.5, η=0.5, ζ=0)
        point = np.array([0.25, 0.75, 0.5])
        candidates = np.array([1])
        eid, xi, eta, zeta = find_containing_element(point, candidates, coords, dxi_dx)
        assert eid == 1
        assert np.isclose(xi, -0.5, atol=1e-6)
        assert np.isclose(eta, 0.5, atol=1e-6)
        assert np.isclose(zeta, 0.0, atol=1e-6)

    def test_not_found_raises(self, synthetic_mesh_path):
        with GeometryReader(synthetic_mesh_path) as gr:
            coords = gr.coords
            dxi_dx = gr.dxi_dx
        point = np.array([5.0, 5.0, 5.0])
        candidates = np.array([1])
        with pytest.raises(ValueError, match="not found"):
            find_containing_element(point, candidates, coords, dxi_dx)

    def test_two_elements(self, synthetic_mesh_2elem_path):
        with GeometryReader(synthetic_mesh_2elem_path) as gr:
            coords = gr.coords
            dxi_dx = gr.dxi_dx
        # Point in element 1
        point = np.array([0.5, 0.5, 0.5])
        candidates = np.array([1, 2])
        eid, xi, eta, zeta = find_containing_element(point, candidates, coords, dxi_dx)
        assert eid == 1
        assert np.isclose(xi, 0.0, atol=1e-6)

        # Point in element 2
        point = np.array([1.5, 0.5, 0.5])
        eid, xi, eta, zeta = find_containing_element(point, candidates, coords, dxi_dx)
        assert eid == 2
        assert np.isclose(xi, 0.0, atol=1e-6)


from gf_post.reader import GeometryReader
