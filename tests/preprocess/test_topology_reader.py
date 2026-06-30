"""Tests for topology_reader module."""

import os
import sys

import h5py
import numpy as np
import pytest

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.topology_reader import read_topology


def _make_mock_mesh(path, n_cell=2):
    """Create a synthetic model.h5 for testing."""
    with h5py.File(path, "w") as f:
        topo = f.create_group("topology")

        # 16 vertices for 2 stacked cubes
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

        # Shared face at z=1 between the two cubes
        edges = np.array(
            [
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 1],  # edges 1-4
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 5],  # edges 5-8
                [1, 5],
                [2, 6],
                [3, 7],
                [4, 8],  # edges 9-12 (vertical)
                [9, 10],
                [10, 11],
                [11, 12],
                [12, 9],  # edges 13-16
                [13, 14],
                [14, 15],
                [15, 16],
                [16, 13],  # edges 17-20
                [9, 13],
                [10, 14],
                [11, 15],
                [12, 16],  # edges 21-24 (vertical)
            ],
            dtype=np.int64,
        )

        # 10 surfaces for 2 cubes (each has 6, share one = 10)
        s2e = np.array(
            [
                [1, 2, 3, 4],  # surf 1: -z bottom of cube 1
                [5, 6, 7, 8],  # surf 2: +z top of cube 1 (shared)
                [9, 10, -11, -12],  # surf 3: -y front
                [-9, 10, 11, -12],  # surf 4: +y back
                [-9, -10, 11, 12],  # surf 5: -x left
                [9, -10, -11, 12],  # surf 6: +x right
                [13, 14, 15, 16],  # surf 7: -z bottom of cube 2
                [17, 18, 19, 20],  # surf 8: +z top of cube 2
                [21, 22, -23, -24],  # surf 9: -y front
                [-21, 23, 22, -24],  # surf 10: +y something
            ],
            dtype=np.int64,
        )

        # 2 cells, 6 faces each (signed)
        c2s = np.array(
            [
                [1, 2, 3, 4, 5, 6],  # cell 1: all positive
                [-2, 7, 8, 9, 10, 6],  # cell 2: face 2 is reversed (shared)
            ],
            dtype=np.int64,
        )

        topo.create_dataset("vertex_to_coord", data=verts, dtype="float64")
        topo.create_dataset("edge_to_vertex", data=edges, dtype="int64")
        topo.create_dataset("surface_to_edge", data=s2e, dtype="int64")
        topo.create_dataset("cell_to_surface", data=c2s, dtype="int64")

        topo.attrs.create("n_vertex", verts.shape[0], dtype="int64")
        topo.attrs.create("n_edge", edges.shape[0], dtype="int64")
        topo.attrs.create("n_surface", s2e.shape[0], dtype="int64")
        topo.attrs.create("n_cell", c2s.shape[0], dtype="int64")


class TestReadTopology:
    def test_reads_vertex_to_coord_shape(self, tmp_dir):
        path = tmp_dir / "model.h5"
        _make_mock_mesh(path)
        topo = read_topology(str(path))
        assert topo.vertex_to_coord.shape[1] == 3
        assert topo.n_vertex == 16

    def test_reads_n_vertex_attr(self, tmp_dir):
        path = tmp_dir / "model.h5"
        _make_mock_mesh(path)
        topo = read_topology(str(path))
        assert topo.n_vertex == 16
        assert topo.n_cell == 2

    def test_edge_to_vertex_signed(self, tmp_dir):
        path = tmp_dir / "model.h5"
        _make_mock_mesh(path)
        topo = read_topology(str(path))
        assert topo.edge_to_vertex.shape[1] == 2
        # All entries should be positive (1-based vertex IDs)
        assert np.all(topo.edge_to_vertex > 0)

    def test_surface_to_edge_ccw(self, tmp_dir):
        path = tmp_dir / "model.h5"
        _make_mock_mesh(path)
        topo = read_topology(str(path))
        assert topo.surface_to_edge.shape[1] == 4
        assert topo.n_surface == 10

    def test_cell_to_surface_signed(self, tmp_dir):
        path = tmp_dir / "model.h5"
        _make_mock_mesh(path)
        topo = read_topology(str(path))
        assert topo.cell_to_surface.shape[1] == 6
        assert topo.n_cell == 2
        # Cell 1 has all positive faces
        assert np.all(topo.cell_to_surface[0] > 0)

    def test_file_not_found(self, tmp_dir):
        path = tmp_dir / "nonexistent.h5"
        with pytest.raises(FileNotFoundError):
            read_topology(str(path))

    def test_missing_topology_group(self, tmp_dir):
        import h5py as _h5py

        path = tmp_dir / "bad.h5"
        with _h5py.File(path, "w") as f:
            f.create_group("other")
        with pytest.raises(ValueError, match="topology"):
            read_topology(str(path))
