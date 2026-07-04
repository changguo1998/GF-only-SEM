"""Tests for gf_post.writer — element-tiled Green's function writing."""

import numpy as np
import h5py
import pytest
from gf_post.writer import GFWriter


# Shared element-grid config for a [0,1]^3 domain with 4x4 elements, no PML,
# and 2 tiles of 2 elements each in x and y.
NX = 4
NY = 4
PML = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}
TILEX = [2, 2]
TILEY = [2, 2]
DOMAIN = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "zmin": 0.0, "zmax": 1.0}


def test_single_tile_writes_six_components(tmp_path):
    """One tile with 2 vertices, 3 timesteps — verify file structure."""
    n_vertex = 2
    nt = 3
    vertex_coords = np.array([[0.0, 0.0, 0.0], [0.3, 0.4, 0.0]], dtype=np.float64)
    vertex_ids = np.array([1, 2], dtype=np.int64)
    time_arr = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    dt = 0.1
    greens = np.zeros((nt, n_vertex, 6, 3), dtype=np.float64)

    tiles = GFWriter.write(
        str(tmp_path / "greenfun"),
        vertex_coords,
        vertex_ids,
        time_arr,
        dt,
        greens,
        nx_elements=NX,
        ny_elements=NY,
        pml_thickness=PML,
        tilex_elements=TILEX,
        tiley_elements=TILEY,
        domain_bounds=DOMAIN,
        record_depth_max_m=1.0,
        record_depth_actual_m=0.5,
    )

    assert len(tiles) == 1  # both vertices in tile (0,0)
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
    """Two vertices in different element bins → two tiles."""
    n_vertex = 2
    nt = 1
    # Vertex 0 at (0,0) → element (0,0) → tile 0
    # Vertex 1 at (0.6,0.7) → element (2,2) → tile 1 (since tile size is 2)
    vertex_coords = np.array([[0.0, 0.0, 0.0], [0.6, 0.7, 0.0]], dtype=np.float64)
    vertex_ids = np.array([1, 2], dtype=np.int64)
    time_arr = np.array([0.0], dtype=np.float64)
    dt = 0.01
    greens = np.zeros((nt, n_vertex, 6, 3), dtype=np.float64)

    tiles = GFWriter.write(
        str(tmp_path / "greenfun"),
        vertex_coords,
        vertex_ids,
        time_arr,
        dt,
        greens,
        nx_elements=NX,
        ny_elements=NY,
        pml_thickness=PML,
        tilex_elements=TILEX,
        tiley_elements=TILEY,
        domain_bounds=DOMAIN,
    )

    assert len(tiles) == 2
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

    with pytest.raises(ValueError, match="greens shape mismatch"):
        GFWriter.write(
            str(tmp_path / "greenfun"),
            vertex_coords,
            vertex_ids,
            time_arr,
            dt,
            greens_wrong,
            nx_elements=NX,
            ny_elements=NY,
            pml_thickness=PML,
            tilex_elements=TILEX,
            tiley_elements=TILEY,
            domain_bounds=DOMAIN,
        )


# ── Spatial tiling tests ──────────────────────────────────────────────


def test_spatial_tiling_single_tile(tmp_path):
    """Spatial tiling: 2 vertices in same spatial bin → 1 tile."""
    n_vertex = 2
    nt = 1
    vertex_coords = np.array([[0.1, 0.2, 0.0], [0.3, 0.4, 0.0]], dtype=np.float64)
    vertex_ids = np.array([1, 2], dtype=np.int64)
    time_arr = np.array([0.0], dtype=np.float64)
    dt = 0.01
    greens = np.zeros((nt, n_vertex, 6, 3), dtype=np.float64)

    tiles = GFWriter.write(
        str(tmp_path / "greenfun"),
        vertex_coords,
        vertex_ids,
        time_arr,
        dt,
        greens,
        nx_elements=NX,
        ny_elements=NY,
        pml_thickness=PML,
        tilex_elements=TILEX,
        tiley_elements=TILEY,
        domain_bounds=DOMAIN,
        green_tile_size_m=0.5,
        record_depth_max_m=1.0,
        record_depth_actual_m=0.5,
    )

    assert len(tiles) == 1  # both vertices in tile (0,0) with green_tile_size_m=0.5
    # Verify tile file is valid HDF5
    with h5py.File(tiles[0], "r") as f:
        assert f.attrs["basis"] == "mesh_vertices"
        assert int(f["time"].attrs["nsteps"]) == 1


def test_spatial_tiling_two_tiles(tmp_path):
    """Spatial tiling: vertices in different spatial bins → 2 tiles."""
    n_vertex = 2
    nt = 1
    # (0.1, 0.1) with green_tile_size_m=0.5 → tile (0,0)
    # (0.6, 0.7) with green_tile_size_m=0.5 → tile (1,1)
    vertex_coords = np.array([[0.1, 0.1, 0.0], [0.6, 0.7, 0.0]], dtype=np.float64)
    vertex_ids = np.array([1, 2], dtype=np.int64)
    time_arr = np.array([0.0], dtype=np.float64)
    dt = 0.01
    greens = np.zeros((nt, n_vertex, 6, 3), dtype=np.float64)

    tiles = GFWriter.write(
        str(tmp_path / "greenfun"),
        vertex_coords,
        vertex_ids,
        time_arr,
        dt,
        greens,
        nx_elements=NX,
        ny_elements=NY,
        pml_thickness=PML,
        tilex_elements=TILEX,
        tiley_elements=TILEY,
        domain_bounds=DOMAIN,
        green_tile_size_m=0.5,
    )

    assert len(tiles) == 2
    tile_names = sorted([p.name for p in tiles])
    assert "tile_x000_y000.h5" in tile_names
    assert "tile_x001_y001.h5" in tile_names


def test_spatial_tiling_fallback_element_mode(tmp_path):
    """When green_tile_size_m is None, falls back to element-count tiling."""
    n_vertex = 2
    nt = 1
    vertex_coords = np.array([[0.0, 0.0, 0.0], [0.6, 0.7, 0.0]], dtype=np.float64)
    vertex_ids = np.array([1, 2], dtype=np.int64)
    time_arr = np.array([0.0], dtype=np.float64)
    dt = 0.01
    greens = np.zeros((nt, n_vertex, 6, 3), dtype=np.float64)

    tiles = GFWriter.write(
        str(tmp_path / "greenfun"),
        vertex_coords,
        vertex_ids,
        time_arr,
        dt,
        greens,
        nx_elements=NX,
        ny_elements=NY,
        pml_thickness=PML,
        tilex_elements=TILEX,
        tiley_elements=TILEY,
        domain_bounds=DOMAIN,
        green_tile_size_m=None,  # explicit None → element-count fallback
    )

    assert len(tiles) == 2  # same behavior as test_two_tiles_splits_by_xy
