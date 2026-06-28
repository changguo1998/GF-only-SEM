"""Tests for gf_post.writer — spatial tile writing."""

import numpy as np
import h5py
from gf_post.writer import GFWriter


def test_single_tile_writes_six_components(tmp_path):
    """One tile with 2 vertices, 3 timesteps — verify file structure."""
    n_vertex = 2
    nt = 3
    vertex_coords = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0]], dtype=np.float64)
    vertex_ids = np.array([1, 2], dtype=np.int64)
    time_arr = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    dt = 0.1
    greens = np.zeros((nt, n_vertex, 6, 3), dtype=np.float64)

    domain_bounds = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}

    tiles = GFWriter.write(
        str(tmp_path / "greenfun"),
        vertex_coords,
        vertex_ids,
        time_arr,
        dt,
        greens,
        green_tile_size_m=0.5,
        domain_bounds=domain_bounds,
        record_depth_max_m=1.0,
        record_depth_actual_m=0.5,
    )

    assert len(tiles) == 1  # both vertices in same tile
    tile_path = tiles[0]

    with h5py.File(tile_path, "r") as f:
        assert f.attrs["version"] == "1.0.0"
        assert f.attrs["basis"] == "mesh_vertices"
        assert f.attrs["record_depth_max_m"] == 1.0
        assert f.attrs["record_depth_actual_m"] == 0.5
        assert f.attrs["excludes_pml"] == True

        assert np.allclose(f["time/t"][:], time_arr)
        assert float(f["time"].attrs["dt"]) == 0.1
        assert int(f["time"].attrs["nsteps"]) == 3

        assert np.array_equal(f["mesh/vertex_ids"][:], vertex_ids)

        g = f["field/greens_tensor"][:]
        assert g.shape == (3, 2, 6, 3)
        assert g.dtype == np.float32


def test_two_tiles_splits_by_xy(tmp_path):
    """Two vertices in different xy bins → two tiles."""
    n_vertex = 2
    nt = 1
    vertex_coords = np.array([[0.0, 0.0, 0.0], [0.6, 0.7, 0.0]], dtype=np.float64)
    vertex_ids = np.array([1, 2], dtype=np.int64)
    time_arr = np.array([0.0], dtype=np.float64)
    dt = 0.01
    greens = np.zeros((nt, n_vertex, 6, 3), dtype=np.float64)
    domain_bounds = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}

    tiles = GFWriter.write(
        str(tmp_path / "greenfun"),
        vertex_coords,
        vertex_ids,
        time_arr,
        dt,
        greens,
        green_tile_size_m=0.5,
        domain_bounds=domain_bounds,
    )

    assert len(tiles) == 2
    # First vertex in tile (0,0), second in tile (1,1)
    tile_names = sorted([p.name for p in tiles])
    assert "tile_x000_y000.h5" in tile_names
    assert "tile_x001_y001.h5" in tile_names


def test_shape_mismatch_raises(tmp_path):
    """Shape mismatch between greens and vertex coords raises ValueError."""
    n_vertex = 2
    nt = 3
    vertex_coords = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0]], dtype=np.float64)
    vertex_ids = np.array([1, 2], dtype=np.int64)
    time_arr = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    dt = 0.1

    # Wrong shape: (nt, n_vertex, 3, 6) instead of (nt, n_vertex, 6, 3)
    greens_wrong = np.zeros((nt, n_vertex, 3, 6), dtype=np.float64)
    domain_bounds = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}

    import pytest

    with pytest.raises(ValueError, match="greens shape mismatch"):
        GFWriter.write(
            str(tmp_path / "greenfun"),
            vertex_coords,
            vertex_ids,
            time_arr,
            dt,
            greens_wrong,
            green_tile_size_m=0.5,
            domain_bounds=domain_bounds,
        )
