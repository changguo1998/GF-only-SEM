"""Tests for SourceRun — one SEM source run with lazy tile loading."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from greenfun.source_run import SourceRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TILE_ATTRS = {
    "version": "1.0.0",
    "basis": "mesh_vertices",
    "tile_y_index": 0,
    "record_depth_max_m": 0.0,
    "record_depth_actual_m": 0.0,
    "excludes_pml": 1,
    "source_xyz_m": np.array([500.0, 0.0, 0.0]),
    "source_directions": "x,y,z",
}


def _regular_grid_coords(
    nx: int,
    ny: int,
    nz: int,
    x0: float = 0.0,
    y0: float = 0.0,
    z0: float = 0.0,
    dx: float = 10.0,
    dy: float = 10.0,
    dz: float = 10.0,
) -> np.ndarray:
    """Return ``(nx*ny*nz, 3)`` array of regularly-spaced 3-D points."""
    xs = x0 + dx * np.arange(nx)
    ys = y0 + dy * np.arange(ny)
    zs = z0 + dz * np.arange(nz)
    mesh = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def _write_tile(
    path: Path,
    vertex_ids: np.ndarray,
    vertex_coords: np.ndarray,
    time: np.ndarray,
    strain: np.ndarray,
    displacement: np.ndarray | None = None,
    tile_x_index: int = 0,
    sem_source_xyz: np.ndarray | None = None,
) -> None:
    """Write a single tile HDF5 file."""
    attrs = dict(_TILE_ATTRS)
    if sem_source_xyz is not None:
        attrs["source_xyz_m"] = sem_source_xyz
    attrs["tile_x_index"] = tile_x_index
    attrs["x_min_m"] = float(vertex_coords[:, 0].min())
    attrs["x_max_m"] = float(vertex_coords[:, 0].max())
    attrs["y_min_m"] = float(vertex_coords[:, 1].min())
    attrs["y_max_m"] = float(vertex_coords[:, 1].max())
    attrs["z_min_m"] = float(vertex_coords[:, 2].min())
    attrs["z_max_m"] = float(vertex_coords[:, 2].max())
    attrs["greens_quantities"] = "strain,displacement" if displacement is not None else "strain"

    with h5py.File(path, "w") as h5:
        for key, value in attrs.items():
            h5.attrs[key] = value

        h5.create_dataset("/time/t", data=time)
        h5.create_dataset("/mesh/vertex_ids", data=vertex_ids)
        h5.create_dataset("/mesh/vertex_coords", data=vertex_coords, dtype=np.float64)
        h5.create_dataset(
            "/field/greens_tensor",
            data=strain,
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        if displacement is not None:
            h5.create_dataset(
                "/field/displacement_tensor",
                data=displacement,
                dtype=np.float32,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )


def _create_source_run_dir(
    tmp_path: Path,
    grid_shape: tuple[int, int, int] = (3, 3, 3),
    nt: int = 50,
    include_displacement: bool = True,
    n_tiles: int = 1,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Create a synthetic source-run directory with a 3-D regular grid.

    Returns ``(dir_path, source_xyz, time)``.
    """
    rng = np.random.default_rng(42)
    time = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    sem_source_xyz = np.array([500.0, 0.0, 0.0])

    src_dir = tmp_path / "src_0000"
    src_dir.mkdir(parents=True)

    nx, ny, nz = grid_shape
    n_vertices = nx * ny * nz

    for tile_idx in range(n_tiles):
        tile_path = src_dir / f"tile_x{tile_idx:03d}_y000.h5"
        shift = tile_idx * 20.0
        coords = _regular_grid_coords(
            nx, ny, nz, x0=0.0 + shift, y0=0.0, z0=0.0, dx=10.0, dy=10.0, dz=10.0
        )

        vids = np.arange(tile_idx * n_vertices, tile_idx * n_vertices + n_vertices, dtype=np.int64)

        strain = rng.standard_normal((nt, n_vertices, 6, 3)).astype(np.float32)

        displacement: np.ndarray | None = None
        if include_displacement:
            displacement = rng.standard_normal((nt, n_vertices, 3, 3)).astype(np.float32)

        _write_tile(
            tile_path,
            vertex_ids=vids,
            vertex_coords=coords,
            time=time,
            strain=strain,
            displacement=displacement,
            tile_x_index=tile_idx,
            sem_source_xyz=sem_source_xyz,
        )

    return src_dir, sem_source_xyz, time

# ---------------------------------------------------------------------------
# Tests — SourceRun.load
# ---------------------------------------------------------------------------


class TestSourceRunLoad:
    """SourceRun.load() correctness."""

    def test_single_tile_loads_correctly(self, tmp_path: Path) -> None:
        """Single-tile SourceRun loads vertex_coords, time, and tensors."""
        src_dir, source_xyz, time = _create_source_run_dir(tmp_path, grid_shape=(3, 3, 3), nt=50)
        run = SourceRun(src_dir, source_xyz)
        assert run._loaded is False, "should be lazy before load()"

        run.load()

        assert run._loaded is True
        assert run.n_tiles == 1
        assert run.time is not None
        assert run.time.shape == (50,)
        assert np.allclose(run.time, time)
        assert run.vertex_coords.shape == (27, 3)  # 3×3×3
        assert run.greens_tensor.shape == (50, 27, 6, 3)
        assert run.displacement_tensor is not None
        assert run.displacement_tensor.shape == (50, 27, 3, 3)

    def test_no_tiles_raises(self, tmp_path: Path) -> None:
        """SourceRun raises FileNotFoundError when no tiles exist."""
        src_dir = tmp_path / "src_0000"
        src_dir.mkdir(parents=True)
        run = SourceRun(src_dir, np.array([0.0, 0.0, 0.0]))
        with pytest.raises(FileNotFoundError, match="No tile"):
            run.load()

    def test_load_displacement_absent(self, tmp_path: Path) -> None:
        """Loading a tile without displacement tensor leaves it None."""
        src_dir, source_xyz, _ = _create_source_run_dir(
            tmp_path, grid_shape=(3, 3, 3), nt=50, include_displacement=False
        )
        run = SourceRun(src_dir, source_xyz)
        run.load()

        assert run.displacement_tensor is None
        assert run.greens_tensor.shape == (50, 27, 6, 3)


# ---------------------------------------------------------------------------
# Tests — SourceRun.query
# ---------------------------------------------------------------------------


class TestSourceRunQuery:
    """SourceRun.query() correctness."""

    @staticmethod
    def _make_run(tmp_path: Path) -> SourceRun:
        src_dir, source_xyz, _ = _create_source_run_dir(tmp_path, grid_shape=(2, 2, 2), nt=50)
        return SourceRun(src_dir, source_xyz)

    def test_query_returns_greenquery_with_correct_shapes(self, tmp_path: Path) -> None:
        """SourceRun query returns GreenQuery with correct shapes."""
        run = self._make_run(tmp_path)
        # (5, 5, 5) is inside the cell [0..10]^3
        result = run.query(source_xyz_m=np.array([5.0, 5.0, 5.0]), quantity="both")

        assert result.source_xyz.shape == (3,)
        assert result.receiver_xyz.shape == (3,)
        assert result.sem_source_xyz.shape == (3,)
        assert np.allclose(result.sem_source_xyz, [500.0, 0.0, 0.0])
        assert result.time.shape == (50,)
        assert result.strain is not None
        assert result.strain.shape == (50, 6, 3)
        assert result.displacement is not None
        assert result.displacement.shape == (50, 3, 3)
        assert result.n_tiles_used == 1
        assert isinstance(result.interpolation_used, bool)

    def test_query_strain_only(self, tmp_path: Path) -> None:
        """Query with quantity='strain' returns only strain."""
        run = self._make_run(tmp_path)
        result = run.query(source_xyz_m=np.array([5.0, 5.0, 5.0]), quantity="strain")

        assert result.strain is not None
        assert result.strain.shape == (50, 6, 3)
        assert result.displacement is None

    def test_query_displacement_only(self, tmp_path: Path) -> None:
        """Query with quantity='displacement' returns only displacement."""
        run = self._make_run(tmp_path)
        result = run.query(source_xyz_m=np.array([5.0, 5.0, 5.0]), quantity="displacement")

        assert result.displacement is not None
        assert result.displacement.shape == (50, 3, 3)
        assert result.strain is None

    def test_displacement_none_when_not_available(self, tmp_path: Path) -> None:
        """Query for displacement when no displacement tensor returns None."""
        src_dir, source_xyz, _ = _create_source_run_dir(
            tmp_path, grid_shape=(2, 2, 2), nt=50, include_displacement=False
        )
        run = SourceRun(src_dir, source_xyz)
        result = run.query(source_xyz_m=np.array([5.0, 5.0, 5.0]), quantity="displacement")
        assert result.displacement is None

    def test_exact_vertex_match_no_interpolation(self, tmp_path: Path) -> None:
        """Exact vertex match returns exact value (no interpolation)."""
        run = self._make_run(tmp_path)
        run.load()

        # Pick the first vertex coordinate.
        vertex = run.vertex_coords[0]
        result = run.query(source_xyz_m=vertex, quantity="strain")

        assert result.interpolation_used is False

        # The result should match the exact stored value at that vertex.
        expected = run.greens_tensor[:, 0, :, :]
        assert result.strain is not None
        assert np.allclose(result.strain, expected), "Exact vertex match returned incorrect value"

    def test_off_vertex_uses_interpolation(self, tmp_path: Path) -> None:
        """Query at a non-vertex coordinate uses interpolation."""
        run = self._make_run(tmp_path)
        result = run.query(source_xyz_m=np.array([5.0, 5.0, 5.0]), quantity="strain")
        assert result.interpolation_used is True

    def test_query_lazy_loads(self, tmp_path: Path) -> None:
        """Query triggers lazy load automatically."""
        run = self._make_run(tmp_path)
        assert run._loaded is False
        result = run.query(source_xyz_m=np.array([5.0, 5.0, 5.0]), quantity="strain")
        assert run._loaded is True
        assert result.strain is not None


# ---------------------------------------------------------------------------
# Tests — Cross-tile vertex deduplication
# ---------------------------------------------------------------------------


class TestSourceRunDeduplication:
    """Cross-tile vertex deduplication."""

    def test_deduplicate_identical_vertex_ids(self, tmp_path: Path) -> None:
        """Tiles sharing identical vertex IDs deduplicate correctly."""
        rng = np.random.default_rng(99)
        time = np.linspace(0.0, 0.5, 30, dtype=np.float64)
        nt = 30
        n_vert_per_tile = 8  # 2×2×2

        src_dir = tmp_path / "src_0000"
        src_dir.mkdir(parents=True)

        # Two tiles with identical vertex_ids (0..7) but different data.
        for tile_idx in range(2):
            tile_path = src_dir / f"tile_x{tile_idx:03d}_y000.h5"
            shift = tile_idx * 20.0
            coords = _regular_grid_coords(
                2, 2, 2, x0=shift, y0=0.0, z0=0.0, dx=10.0, dy=10.0, dz=10.0
            )
            vids = np.arange(n_vert_per_tile, dtype=np.int64)
            strain = rng.standard_normal((nt, n_vert_per_tile, 6, 3)).astype(np.float32)
            displacement = rng.standard_normal((nt, n_vert_per_tile, 3, 3)).astype(np.float32)

            _write_tile(
                tile_path,
                vertex_ids=vids,
                vertex_coords=coords,
                time=time,
                strain=strain,
                displacement=displacement,
                tile_x_index=tile_idx,
            )

        run = SourceRun(src_dir, np.array([500.0, 0.0, 0.0]))
        run.load()

        # Should have 8 unique vertices (first tile's copy).
        assert run.vertex_coords.shape == (8, 3)
        assert run.n_tiles == 2
        assert run.greens_tensor.shape == (nt, 8, 6, 3)
        assert run.displacement_tensor is not None
        assert run.displacement_tensor.shape == (nt, 8, 3, 3)

    def test_deduplicate_partial_overlap(self, tmp_path: Path) -> None:
        """Tiles with partially overlapping vertex IDs deduplicate correctly.

        Tile 0: vertex_ids [0, 1, 2, 3, 4]
        Tile 1: vertex_ids [3, 4, 5, 6, 7]  (3,4 overlap)
        Result: 8 unique vertices (first tile's copy of 3,4 kept)
        """
        rng = np.random.default_rng(101)
        time = np.linspace(0.0, 0.5, 20, dtype=np.float64)
        nt = 20

        src_dir = tmp_path / "src_0000"
        src_dir.mkdir(parents=True)

        # Tile 0: ids 0-4
        tile0_path = src_dir / "tile_x000_y000.h5"
        coords0 = _regular_grid_coords(5, 1, 1, x0=0.0, y0=0.0, z0=0.0, dx=10.0)[:5]
        vids0 = np.arange(5, dtype=np.int64)
        strain0 = rng.standard_normal((nt, 5, 6, 3)).astype(np.float32)
        disp0 = rng.standard_normal((nt, 5, 3, 3)).astype(np.float32)
        _write_tile(tile0_path, vids0, coords0, time, strain0, disp0, tile_x_index=0)

        # Tile 1: ids 3-7 (overlaps with 3,4)
        tile1_path = src_dir / "tile_x001_y000.h5"
        coords1 = _regular_grid_coords(5, 1, 1, x0=0.0, y0=0.0, z0=0.0, dx=10.0)[:5]
        vids1 = np.arange(3, 8, dtype=np.int64)
        strain1 = rng.standard_normal((nt, 5, 6, 3)).astype(np.float32)
        disp1 = rng.standard_normal((nt, 5, 3, 3)).astype(np.float32)
        _write_tile(tile1_path, vids1, coords1, time, strain1, disp1, tile_x_index=1)

        run = SourceRun(src_dir, np.array([500.0, 0.0, 0.0]))
        run.load()

        # 8 unique vertices: ids [0,1,2,3,4,5,6,7]
        assert run.vertex_coords.shape == (8, 3)
        assert run.greens_tensor.shape == (nt, 8, 6, 3)
        assert run.displacement_tensor.shape == (nt, 8, 3, 3)

        # First tile's coords at positions 0-4 should be preserved.
        assert np.allclose(run.vertex_coords[0], coords0[0])
        assert np.allclose(run.vertex_coords[4], coords0[4])

        # Tile 1's unique vertices (5,6,7) should be appended.
        assert np.allclose(run.vertex_coords[5], coords1[2])
        assert np.allclose(run.vertex_coords[6], coords1[3])
        assert np.allclose(run.vertex_coords[7], coords1[4])

    def test_deduplicate_three_tiles(self, tmp_path: Path) -> None:
        """Three tiles with various overlaps deduplicate correctly."""
        rng = np.random.default_rng(202)
        time = np.linspace(0.0, 0.5, 15, dtype=np.float64)
        nt = 15

        src_dir = tmp_path / "src_0000"
        src_dir.mkdir(parents=True)

        tile_specs = [
            (np.arange(0, 5), 0),  # ids 0,1,2,3,4
            (np.arange(4, 9), 1),  # ids 4,5,6,7,8 -> overlap on 4
            (np.arange(8, 11), 2),  # ids 8,9,10 -> overlap on 8
        ]

        for vids, tile_idx in tile_specs:
            tile_path = src_dir / f"tile_x{tile_idx:03d}_y000.h5"
            nv = len(vids)
            coords = _regular_grid_coords(nv, 1, 1, x0=0.0, y0=0.0, z0=0.0, dx=10.0)[:nv]
            strain = rng.standard_normal((nt, nv, 6, 3)).astype(np.float32)
            _write_tile(
                tile_path,
                vertex_ids=vids,
                vertex_coords=coords,
                time=time,
                strain=strain,
                tile_x_index=tile_idx,
            )

        run = SourceRun(src_dir, np.array([500.0, 0.0, 0.0]))
        run.load()

        # 11 unique vertices: [0,1,2,3,4,5,6,7,8,9,10]
        assert run.vertex_coords.shape == (11, 3)
        assert run.greens_tensor.shape == (nt, 11, 6, 3)
        assert run.n_tiles == 3


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------


class TestSourceRunEdgeCases:
    """SourceRun edge cases."""

    def test_query_wrong_shape_raises(self, tmp_path: Path) -> None:
        """Query with wrong-shaped point raises ValueError."""
        src_dir, source_xyz, _ = _create_source_run_dir(tmp_path, grid_shape=(2, 2, 2), nt=50)
        run = SourceRun(src_dir, source_xyz)
        with pytest.raises(ValueError, match="shape"):
            run.query(np.array([1.0, 2.0]))

    def test_all_vertices_shared_skipped(self, tmp_path: Path) -> None:
        """A tile whose vertices are all duplicates is entirely skipped."""
        rng = np.random.default_rng(303)
        time = np.linspace(0.0, 0.5, 10, dtype=np.float64)
        nt = 10

        src_dir = tmp_path / "src_0000"
        src_dir.mkdir(parents=True)

        # Tile 0: 8 vertices, ids 0-7 (2×2×2 grid)
        tile0_path = src_dir / "tile_x000_y000.h5"
        coords0 = _regular_grid_coords(2, 2, 2, x0=0.0, dx=10.0, dy=10.0, dz=10.0)
        strain0 = rng.standard_normal((nt, 8, 6, 3)).astype(np.float32)
        _write_tile(tile0_path, np.arange(8), coords0, time, strain0, tile_x_index=0)

        # Tile 1: same 8 vertex IDs, different data
        tile1_path = src_dir / "tile_x001_y000.h5"
        coords1 = _regular_grid_coords(2, 2, 2, x0=20.0, dx=10.0, dy=10.0, dz=10.0)
        strain1 = rng.standard_normal((nt, 8, 6, 3)).astype(np.float32)
        _write_tile(tile1_path, np.arange(8), coords1, time, strain1, tile_x_index=1)

        run = SourceRun(src_dir, np.array([500.0, 0.0, 0.0]))
        run.load()

        # Only tile 0's vertices appear.
        assert run.vertex_coords.shape == (8, 3)
        assert run.n_tiles == 2
        assert np.allclose(run.vertex_coords, coords0)
