"""Tests for gf_post.geometry — GLL nodes, weights, and Lagrange basis."""

import numpy as np
import pytest

from gf_post.geometry import (
    gll_nodes_1d,
    gll_weights_1d,
    gll_nodes_3d,
    lagrange_basis_1d,
    lagrange_basis_3d,
)


class TestGLLNodes1D:
    def test_n1_shape(self):
        assert gll_nodes_1d(1).shape == (2,)

    def test_n2_shape(self):
        assert gll_nodes_1d(2).shape == (3,)

    def test_n5_shape(self):
        assert gll_nodes_1d(5).shape == (6,)

    def test_endpoints(self):
        for N in range(1, 6):
            pts = gll_nodes_1d(N)
            assert np.isclose(pts[0], -1.0)
            assert np.isclose(pts[-1], 1.0)

    def test_in_range(self):
        for N in range(1, 6):
            pts = gll_nodes_1d(N)
            assert np.all(pts >= -1.0 - 1e-10)
            assert np.all(pts <= 1.0 + 1e-10)

    def test_symmetric(self):
        for N in range(2, 6):
            pts = gll_nodes_1d(N)
            assert np.allclose(pts, -pts[::-1])

    def test_ordered(self):
        for N in range(1, 6):
            pts = gll_nodes_1d(N)
            assert np.all(np.diff(pts) > 0)


class TestGLLWeights1D:
    def test_positive(self):
        for N in range(1, 6):
            w = gll_weights_1d(N)
            assert np.all(w > 0)

    def test_sum_to_2(self):
        for N in range(1, 6):
            w = gll_weights_1d(N)
            assert np.isclose(w.sum(), 2.0, atol=1e-10)

    def test_shape(self):
        for N in range(1, 6):
            assert gll_weights_1d(N).shape == (N + 1,)


class TestGLLNodes3D:
    def test_shape(self):
        for N in range(1, 4):
            ngll = N + 1
            nodes = gll_nodes_3d(N)
            assert nodes.shape == (ngll, ngll, ngll, 3)

    def test_unit_cube_coords(self):
        """For unit cube centered at origin, 3D nodes should be in [-1,1]^3."""
        nodes = gll_nodes_3d(1)  # N=1, ngll=2
        for d in range(3):
            assert np.all(nodes[:, :, :, d] >= -1 - 1e-10)
            assert np.all(nodes[:, :, :, d] <= 1 + 1e-10)


class TestLagrangeBasis1D:
    def test_kronecker(self):
        """L_i(xi_j) = delta_ij."""
        for N in range(1, 6):
            nodes = gll_nodes_1d(N)
            ngll = len(nodes)
            for j in range(ngll):
                basis = lagrange_basis_1d(nodes[j], nodes)
                assert basis.shape == (ngll,)
                expected = np.zeros(ngll)
                expected[j] = 1.0
                assert np.allclose(basis, expected, atol=1e-10)

    def test_sum_to_one(self):
        """Sum of Lagrange basis at any point should be 1 (partition of unity)."""
        nodes = gll_nodes_1d(3)
        for xi in [-0.5, 0.0, 0.5, 0.999, -0.999]:
            basis = lagrange_basis_1d(xi, nodes)
            assert np.isclose(basis.sum(), 1.0, atol=1e-10)


class TestLagrangeBasis3D:
    def test_kronecker_3d(self):
        """3D basis at GLL nodes gives unit tensor."""
        nodes_1d = gll_nodes_1d(2)
        ngll = len(nodes_1d)
        basis = lagrange_basis_3d((nodes_1d[1], nodes_1d[1], nodes_1d[1]), nodes_1d)
        assert basis.shape == (ngll, ngll, ngll)
        assert np.isclose(basis[1, 1, 1], 1.0, atol=1e-10)
        mask = np.zeros((ngll, ngll, ngll), dtype=bool)
        mask[1, 1, 1] = True
        assert np.isclose(basis[~mask].max(), 0.0, atol=1e-10)

    def test_partition_of_unity(self):
        nodes_1d = gll_nodes_1d(2)
        basis = lagrange_basis_3d((0.0, 0.0, 0.0), nodes_1d)
        assert np.isclose(basis.sum(), 1.0, atol=1e-10)

    def test_shape(self):
        for N in range(1, 4):
            nodes_1d = gll_nodes_1d(N)
            ngll = len(nodes_1d)
            basis = lagrange_basis_3d((0.0, 0.0, 0.0), nodes_1d)
            assert basis.shape == (ngll, ngll, ngll)