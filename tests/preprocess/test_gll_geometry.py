import os
import sys

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

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
        import meshio

        from preprocess.topology_reader import TopologyData
        from tools.gmsh_to_hdf5 import extract_topology

        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float64,
        )
        hex_cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        cells = [("hexahedron", hex_cells)]
        mesh = meshio.Mesh(vertices, cells)
        topo_dict = extract_topology(mesh)
        return TopologyData(
            vertex_to_coord=topo_dict["vertex_to_coord"],
            edge_to_vertex=topo_dict["edge_to_vertex"],
            surface_to_edge=topo_dict["surface_to_edge"],
            cell_to_surface=topo_dict["cell_to_surface"],
            n_vertex=8,
            n_edge=12,
            n_surface=6,
            n_cell=1,
        )

    def test_cube_corners_n1(self):
        topo = self._make_unit_cube_topo()
        coords, _, _, _ = compute_gll_geometry(topo, N=1)
        assert np.isclose(coords[0, 0, 0, 0, 0], 0.0)
        assert np.isclose(coords[0, 1, 1, 1, 0], 1.0)

    def test_jacobian_det(self):
        topo = self._make_unit_cube_topo()
        _, jac, _, _ = compute_gll_geometry(topo, N=1)
        assert np.isclose(jac[0, 0, 0, 0], 0.125, atol=1e-10)
        assert np.isclose(jac[0, 1, 1, 1], 0.125, atol=1e-10)

    def test_mass_sum(self):
        topo = self._make_unit_cube_topo()
        _, _, _, mass = compute_gll_geometry(topo, N=1)
        assert np.isclose(mass.sum(), 1.0, atol=1e-10)


class TestMultiElementGLLGeometry:
    """Tests for multi-element meshes where sorted vertex order ≠ GMSH order."""

    def _make_two_element_mesh(self):
        """Create a 2x1x1 hex mesh (nx=2, ny=1, nz=1, lx=2, ly=1, lz=1).

        Element 0: x in [0,1], y in [0,1], z in [0,1]
        Element 1: x in [1,2], y in [0,1], z in [0,1]

        For element 0, GMSH vertex order (1-based):
          [1, 2, 5, 4, 7, 8, 11, 10]
        Sorted order would be:
          [1, 2, 4, 5, 7, 8, 10, 11]
        The sorted order swaps vertices 4↔5 and 10↔11, giving wrong
        physical coordinates for the +y face corners.
        """
        import meshio

        from preprocess.topology_reader import TopologyData
        from tools.gmsh_to_hdf5 import extract_topology

        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],  # x=0,1,2; y=0; z=0
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],  # x=0,1,2; y=1; z=0
                [0, 0, 1],
                [1, 0, 1],
                [2, 0, 1],  # x=0,1,2; y=0; z=1
                [0, 1, 1],
                [1, 1, 1],
                [2, 1, 1],  # x=0,1,2; y=1; z=1
            ],
            dtype=np.float64,
        )
        # Element 0: vertices 0,1,4,3,6,7,10,9
        # Element 1: vertices 1,2,5,4,7,8,11,10
        hex_cells = np.array(
            [[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]], dtype=np.int64
        )
        cells = [("hexahedron", hex_cells)]
        mesh = meshio.Mesh(vertices, cells)
        topo_dict = extract_topology(mesh)
        return TopologyData(
            vertex_to_coord=topo_dict["vertex_to_coord"],
            edge_to_vertex=topo_dict["edge_to_vertex"],
            surface_to_edge=topo_dict["surface_to_edge"],
            cell_to_surface=topo_dict["cell_to_surface"],
            n_vertex=12,
            n_edge=20,
            n_surface=11,
            n_cell=2,
        )

    def test_element0_corners_correct_order(self):
        """Element 0 corners must be in GMSH order, not sorted."""
        topo = self._make_two_element_mesh()
        coords, _, _, _ = compute_gll_geometry(topo, N=1)  # NGLL=2

        # Element 0, N=1: coords[0, i, j, k] = vertex at GLL position (i,j,k)
        # where i,j,k ∈ {0,1} mapping to xi,eta,zeta ∈ {-1, +1}.
        # Verify all 8 corners match expected physical coordinates:

        # v0: xi=-1, eta=-1, zeta=-1 → (0,0,0)
        np.testing.assert_allclose(coords[0, 0, 0, 0], [0.0, 0.0, 0.0], atol=1e-12)
        # v1: xi=+1, eta=-1, zeta=-1 → (1,0,0)
        np.testing.assert_allclose(coords[0, 1, 0, 0], [1.0, 0.0, 0.0], atol=1e-12)
        # v2: xi=+1, eta=+1, zeta=-1 → (1,1,0)
        np.testing.assert_allclose(coords[0, 1, 1, 0], [1.0, 1.0, 0.0], atol=1e-12)
        # v3: xi=-1, eta=+1, zeta=-1 → (0,1,0)
        np.testing.assert_allclose(coords[0, 0, 1, 0], [0.0, 1.0, 0.0], atol=1e-12)
        # v4: xi=-1, eta=-1, zeta=+1 → (0,0,1)
        np.testing.assert_allclose(coords[0, 0, 0, 1], [0.0, 0.0, 1.0], atol=1e-12)
        # v5: xi=+1, eta=-1, zeta=+1 → (1,0,1)
        np.testing.assert_allclose(coords[0, 1, 0, 1], [1.0, 0.0, 1.0], atol=1e-12)
        # v6: xi=+1, eta=+1, zeta=+1 → (1,1,1)
        np.testing.assert_allclose(coords[0, 1, 1, 1], [1.0, 1.0, 1.0], atol=1e-12)
        # v7: xi=-1, eta=+1, zeta=+1 → (0,1,1)
        np.testing.assert_allclose(coords[0, 0, 1, 1], [0.0, 1.0, 1.0], atol=1e-12)

    def test_element1_corners_correct_order(self):
        """Element 1 corners must be in GMSH order, not sorted."""
        topo = self._make_two_element_mesh()
        coords, _, _, _ = compute_gll_geometry(topo, N=1)

        # Element 1: x∈[1,2]
        np.testing.assert_allclose(coords[1, 0, 0, 0], [1.0, 0.0, 0.0], atol=1e-12)  # v0
        np.testing.assert_allclose(coords[1, 1, 0, 0], [2.0, 0.0, 0.0], atol=1e-12)  # v1
        np.testing.assert_allclose(coords[1, 1, 1, 0], [2.0, 1.0, 0.0], atol=1e-12)  # v2
        np.testing.assert_allclose(coords[1, 0, 1, 0], [1.0, 1.0, 0.0], atol=1e-12)  # v3
        np.testing.assert_allclose(coords[1, 0, 0, 1], [1.0, 0.0, 1.0], atol=1e-12)  # v4
        np.testing.assert_allclose(coords[1, 1, 0, 1], [2.0, 0.0, 1.0], atol=1e-12)  # v5
        np.testing.assert_allclose(coords[1, 1, 1, 1], [2.0, 1.0, 1.0], atol=1e-12)  # v6
        np.testing.assert_allclose(coords[1, 0, 1, 1], [1.0, 1.0, 1.0], atol=1e-12)  # v7

    def test_jacobian_consistent_across_elements(self):
        """Both elements should have same Jacobian (identical unit cubes)."""
        topo = self._make_two_element_mesh()
        _, jac, _, _ = compute_gll_geometry(topo, N=1)

        np.testing.assert_allclose(jac[0], jac[1], atol=1e-12)
        assert np.isclose(jac[0, 0, 0, 0], 0.125, atol=1e-10)

    def test_mass_sum_per_element(self):
        """Each element's mass should sum to the element volume (1.0)."""
        topo = self._make_two_element_mesh()
        _, _, _, mass = compute_gll_geometry(topo, N=1)

        assert np.isclose(mass[0].sum(), 1.0, atol=1e-10)
        assert np.isclose(mass[1].sum(), 1.0, atol=1e-10)
