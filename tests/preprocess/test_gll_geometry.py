import os, sys
_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

import pytest
import numpy as np
from preprocess.gll_geometry import compute_gll_geometry, gll_quadrature_points, gll_weights


class TestGLLQuadrature:
    def test_shapes(self):
        assert gll_quadrature_points(1).shape == (2,)
        assert gll_quadrature_points(2).shape == (3,)
        assert gll_quadrature_points(3).shape == (4,)

    def test_range(self):
        for N in range(1, 6):
            pts = gll_quadrature_points(N)
            assert np.all(pts >= -1 - 1e-10) and np.all(pts <= 1 + 1e-10)

    def test_endpoints(self):
        for N in range(1, 6):
            pts = gll_quadrature_points(N)
            assert np.isclose(pts[0], -1.0) and np.isclose(pts[-1], 1.0)

    def test_symmetric(self):
        for N in range(2, 6):
            pts = gll_quadrature_points(N)
            assert np.allclose(pts, -pts[::-1])


class TestGLLWeights:
    def test_positive(self):
        for N in range(1, 5):
            pts = gll_quadrature_points(N)
            w = gll_weights(pts, N)
            assert np.all(w > 0)

    def test_sum_to_2(self):
        for N in range(1, 5):
            pts = gll_quadrature_points(N)
            w = gll_weights(pts, N)
            assert np.isclose(w.sum(), 2.0, atol=1e-10)


class TestGLLGeometry:
    def _make_unit_cube_topo(self):
        from preprocess.topology_reader import TopologyData
        verts = np.array([
            [0,0,0],[1,0,0],[1,1,0],[0,1,0],
            [0,0,1],[1,0,1],[1,1,1],[0,1,1],
        ], dtype=np.float64)
        e2v = np.zeros((12,2), dtype=np.int64)
        s2e = np.zeros((6,4), dtype=np.int64)
        e2v[0]=[1,2]; e2v[1]=[2,3]; e2v[2]=[3,4]; e2v[3]=[4,1]
        e2v[4]=[5,6]; e2v[5]=[6,7]; e2v[6]=[7,8]; e2v[7]=[8,5]
        e2v[8]=[1,5]; e2v[9]=[2,6]; e2v[10]=[3,7]; e2v[11]=[4,8]
        s2e[0]=[1,2,3,4]; s2e[1]=[5,6,7,8]
        s2e[2]=[9,-11,-10,12]; s2e[3]=[10,-12,-9,11]
        s2e[4]=[9,10,-11,-12]; s2e[5]=[-9,11,-10,12]
        c2s = np.array([[1,2,3,4,5,6]], dtype=np.int64)
        return TopologyData(verts, e2v, s2e, c2s, 8, 12, 6, 1)

    def test_cube_corners_n1(self):
        topo = self._make_unit_cube_topo()
        coords, _, _, _ = compute_gll_geometry(topo, N=1)
        assert np.isclose(coords[0,0,0,0,0], 0.0)
        assert np.isclose(coords[0,1,1,1,0], 1.0)

    def test_jacobian_det(self):
        topo = self._make_unit_cube_topo()
        _, jac, _, _ = compute_gll_geometry(topo, N=1)
        assert np.isclose(jac[0,0,0,0], 0.125, atol=1e-10)
        assert np.isclose(jac[0,1,1,1], 0.125, atol=1e-10)

    def test_mass_sum(self):
        topo = self._make_unit_cube_topo()
        _, _, _, mass = compute_gll_geometry(topo, N=1)
        assert np.isclose(mass.sum(), 1.0, atol=1e-10)