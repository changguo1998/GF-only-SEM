"""Unit tests for gmsh_to_hdf5 topology extraction, HDF5 I/O, and auxiliary."""

import os
import sys
import tempfile

import h5py
import meshio
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.gmsh_to_hdf5 import _same_orientation, extract_topology, write_auxiliary, write_topology


def make_mesh(hex_cells, points=None):
    """Helper: create a meshio.Mesh from a list of hex cell vertex arrays."""
    if points is None:
        # Collect unique vertices from hex cells
        all_verts = set()
        for h in hex_cells:
            all_verts.update(int(v) for v in h)
        # Determine max vertex index to build points array
        max_v = max(all_verts)
        # Generate random-ish positions
        np.random.seed(42)
        pts = np.random.rand(max_v + 1, 3)
    else:
        pts = np.asarray(points, dtype=float)

    cells = [("hexahedron", np.array(hex_cells, dtype=np.int64))]
    return meshio.Mesh(pts, cells)


class TestEdgeDeduplication:
    def test_single_hex_12_edges(self):
        """A single hexahedron produces exactly 12 unique edges."""
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo = extract_topology(mesh)
        assert topo["edge_to_vertex"].shape == (12, 2)

    def test_two_stacked_hexes_20_edges(self):
        """Two stacked hexes share the middle 2x2 face -> 12+12-4 = 20 edges."""
        cells = [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]]
        mesh = make_mesh(cells)
        topo = extract_topology(mesh)
        assert topo["edge_to_vertex"].shape == (20, 2)

    def test_edge_canonical_orientation(self):
        """Edge stores (v_low, v_high) with v_low < v_high."""
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
            dtype=float,
        )
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]], points=vertices)
        topo = extract_topology(mesh)
        e2v = topo["edge_to_vertex"]
        for row in e2v:
            assert row[0] < row[1], f"Edge {row} not canonical: v1 < v2 required"

    def test_edge_no_zeros(self):
        """No zeros in edge_to_vertex (1-based indexing, 0 = null)."""
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo = extract_topology(mesh)
        assert np.all(topo["edge_to_vertex"] > 0)


class TestSurfaceExtraction:
    def test_single_hex_6_surfaces(self):
        """A single hex produces exactly 6 surfaces."""
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo = extract_topology(mesh)
        assert topo["surface_to_edge"].shape == (6, 4)

    def test_surface_edges_are_valid_edge_ids(self):
        """All surface edge entries reference valid 1-based edge IDs."""
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo = extract_topology(mesh)
        n_edge = topo["edge_to_vertex"].shape[0]
        abs_edges = np.abs(topo["surface_to_edge"])
        assert np.all(abs_edges >= 1)
        assert np.all(abs_edges <= n_edge)

    def test_two_hexes_11_surfaces(self):
        """Two stacked hexes share 1 interior face: 6+6-1=11 surfaces."""
        cells = [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]]
        mesh = make_mesh(cells)
        topo = extract_topology(mesh)
        assert topo["surface_to_edge"].shape == (11, 4)


class TestCellToSurface:
    def test_single_hex_all_positive(self):
        """For a single hex, all 6 surface references are positive (outward)."""
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo = extract_topology(mesh)
        assert np.all(topo["cell_to_surface"] > 0)

    def test_interior_face_opposite_signs(self):
        """Interior face: + for one cell, - for the other."""
        cells = [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]]
        mesh = make_mesh(cells)
        topo = extract_topology(mesh)
        c2s = topo["cell_to_surface"]

        # Cell 0 top face = index 1, Cell 1 bottom face = index 0
        s0 = c2s[0, 1]
        s1 = c2s[1, 0]
        assert abs(s0) == abs(s1), "Interior surfaces should match"
        assert s0 != s1, "Interior surface signs should differ"

    def test_each_cell_has_6_surfaces(self):
        """Every cell has exactly 6 surfaces."""
        cells = [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]]
        mesh = make_mesh(cells)
        topo = extract_topology(mesh)
        assert topo["cell_to_surface"].shape[1] == 6


class TestWriteTopology:
    def test_write_and_read_back(self):
        """Write topology to HDF5, read back and verify contents."""
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo = extract_topology(mesh)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.h5")
            write_topology(path, topo)

            with h5py.File(path, "r") as f:
                g = f["topology"]
                # Attributes
                assert g.attrs["n_vertex"] == 8
                assert g.attrs["n_edge"] == 12
                assert g.attrs["n_surface"] == 6
                assert g.attrs["n_cell"] == 1

                # Datasets
                assert g["vertex_to_coord"].shape == (8, 3)
                assert g["vertex_to_coord"].dtype == np.float64
                assert g["edge_to_vertex"].shape == (12, 2)
                assert g["edge_to_vertex"].dtype == np.int64
                assert g["surface_to_edge"].shape == (6, 4)
                assert g["surface_to_edge"].dtype == np.int64
                assert g["cell_to_surface"].shape == (1, 6)
                assert g["cell_to_surface"].dtype == np.int64

    def test_no_zeros_in_1based_indices(self):
        """Verify no zero values in 1-based index datasets."""
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo = extract_topology(mesh)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "model.h5")
            write_topology(path, topo)

            with h5py.File(path, "r") as f:
                g = f["topology"]
                assert np.all(g["edge_to_vertex"][:] > 0)
                assert np.all(np.abs(g["surface_to_edge"][:]) > 0)
                assert np.all(np.abs(g["cell_to_surface"][:]) > 0)


class TestWriteAuxiliary:
    def test_surface_to_cell_interior_2_entries(self):
        """Interior surfaces have 2 adjacent cells."""
        cells = [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 8, 9, 10, 11]]
        mesh = make_mesh(cells)
        topo = extract_topology(mesh)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "aux.h5")
            write_auxiliary(path, topo)

            with h5py.File(path, "r") as f:
                s2c = f["auxiliary/surface_to_cell"][:]

        # Interior surface has 2 entries
        interior_ids = [i for i in range(len(s2c)) if s2c[i, 1] != 0]
        assert len(interior_ids) == 1, "Expected 1 interior surface"

    def test_surface_to_cell_boundary_1_entry(self):
        """Boundary surfaces have exactly 1 adjacent cell."""
        cells = [[0, 1, 2, 3, 4, 5, 6, 7]]
        mesh = make_mesh(cells)
        topo = extract_topology(mesh)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "aux.h5")
            write_auxiliary(path, topo)

            with h5py.File(path, "r") as f:
                s2c = f["auxiliary/surface_to_cell"][:]

        # All are boundary surfaces
        for row in s2c:
            assert row[0] != 0, "Each surface should have at least 1 cell"
            assert row[1] == 0, "Boundary surfaces should have only 1 cell"

    def test_vertex_to_edge_csr_shape(self):
        """CSR matrices have correct structure."""
        mesh = make_mesh([[0, 1, 2, 3, 4, 5, 6, 7]])
        topo = extract_topology(mesh)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "aux.h5")
            write_auxiliary(path, topo)

            with h5py.File(path, "r") as f:
                v2e = f["auxiliary/vertex_to_edge"]
                assert v2e["indptr"].shape[0] == 9  # n_vertex + 1 = 8 + 1
                assert v2e["indices"].shape[0] == 24  # 12 edges * 2 ends


class TestSameOrientation:
    def test_same_orientation_identical(self):
        sa = [1, 2, -3, 4]
        sb = [1, 2, -3, 4]
        assert _same_orientation(sa, sb)

    def test_same_orientation_rotated(self):
        sa = [1, 2, -3, 4]
        sb = [-3, 4, 1, 2]
        assert _same_orientation(sa, sb)

    def test_opposite_orientation(self):
        sa = [1, 2, -3, 4]
        sb = [-4, 3, -2, -1]  # reversed
        assert not _same_orientation(sa, sb)
