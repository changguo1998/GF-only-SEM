"""Tests for GreenFunctionLibrary — top-level entry point for reciprocity queries."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from greenfun.library import GreenFunctionLibrary


# ---------------------------------------------------------------------------
# Helpers — tile writer (mirrors test_source_run for independence)
# ---------------------------------------------------------------------------

_TILE_ATTRS = {
    "version": "1.0.0",
    "basis": "mesh_vertices",
    "tile_y_index": 0,
    "record_depth_max_m": 0.0,
    "record_depth_actual_m": 0.0,
    "excludes_pml": 1,
    "source_directions": "x,y,z",
}


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


# ---------------------------------------------------------------------------
# Local fixtures — 3D library for interpolation tests
# ---------------------------------------------------------------------------


def _create_3d_library(
    root: Path,
    n_sources: int = 2,
    n_tiles: int = 1,
    grid_shape: tuple[int, int, int] = (2, 2, 2),
    nt: int = 50,
    include_displacement: bool = True,
) -> None:
    """Create a synthetic library with 3-D vertex data."""
    rng = np.random.default_rng(42)

    rng = np.random.default_rng(42)
    time = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    n_vert = grid_shape[0] * grid_shape[1] * grid_shape[2]

    # SEM sources along the x-axis.
    sem_sources = np.column_stack(
        [np.linspace(0.0, 1000.0, n_sources), np.zeros(n_sources), np.zeros(n_sources)]
    )

    for src_idx in range(n_sources):
        src_dir = root / f"src_{src_idx:04d}"
        src_dir.mkdir(parents=True)
        sem_xyz = sem_sources[src_idx]

        for tile_idx in range(n_tiles):
            tile_path = src_dir / f"tile_x{tile_idx:03d}_y000.h5"
            shift = tile_idx * 20.0

            # 3D regular grid.
            nx, ny, nz = grid_shape
            xs = np.linspace(shift, shift + 10.0 * (nx - 1), nx)
            ys = np.linspace(0.0, 10.0 * (ny - 1), ny)
            zs = np.linspace(0.0, 10.0 * (nz - 1), nz)
            mesh = np.meshgrid(xs, ys, zs, indexing="ij")
            vertex_coords = np.column_stack([m.ravel() for m in mesh])

            vertex_ids = np.arange(tile_idx * n_vert, tile_idx * n_vert + n_vert, dtype=np.int64)

            strain = rng.standard_normal((nt, n_vert, 6, 3)).astype(np.float32)

            displacement: np.ndarray | None = None
            if include_displacement:
                displacement = rng.standard_normal((nt, n_vert, 3, 3)).astype(np.float32)

            _write_tile(
                tile_path,
                vertex_ids=vertex_ids,
                vertex_coords=vertex_coords,
                time=time,
                strain=strain,
                displacement=displacement,
                tile_x_index=tile_idx,
                sem_source_xyz=sem_xyz,
            )


@pytest.fixture
def library_3d(tmp_path: Path) -> Path:
    """Create a 3-D synthetic library and return the root path."""
    _create_3d_library(tmp_path, n_sources=2, n_tiles=1, grid_shape=(3, 3, 3), nt=50)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests — library construction and indexing
# ---------------------------------------------------------------------------


class TestGreenFunctionLibraryInit:
    """GreenFunctionLibrary initialisation and source indexing."""

    def test_indexes_sources_correctly(self, greenfun_library) -> None:
        """Library indexes 3 sources with correct metadata."""
        root = greenfun_library.root
        lib = GreenFunctionLibrary(root)

        assert lib.n_sources == 3
        assert lib.n_tiles == 6  # 3 sources × 2 tiles each

    def test_n_sources_zero_for_empty_library(self, tmp_path: Path) -> None:
        """Empty library has zero sources and zero tiles."""
        lib = GreenFunctionLibrary(tmp_path)
        assert lib.n_sources == 0
        assert lib.n_tiles == 0

    def test_rebuild_index_flag_works(self, greenfun_library) -> None:
        """Passing rebuild_index=True does not cause errors."""
        root = greenfun_library.root
        lib = GreenFunctionLibrary(root, rebuild_index=True)
        assert lib.n_sources == 3

    def test_sources_have_correct_sem_xyz(self, greenfun_library) -> None:
        """SEM source coordinates match the fixture placements."""
        root = greenfun_library.root
        lib = GreenFunctionLibrary(root)

        # The fixture places 3 sources at (0,0,0), (500,0,0), (1000,0,0).
        expected = np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0], [1000.0, 0.0, 0.0]])
        assert np.allclose(lib._source_xyz_array, expected)


# ---------------------------------------------------------------------------
# Tests — single query
# ---------------------------------------------------------------------------


class TestGreenFunctionLibraryQuery:
    """GreenFunctionLibrary.query() correctness."""

    def test_query_returns_greenquery_with_correct_shapes(self, library_3d) -> None:
        """Single query returns GreenQuery with correct shape fields."""
        lib = GreenFunctionLibrary(library_3d)

        # receiver near source 0 (at 0,0,0) → source 0
        # source_xyz inside the first tile's 3-D cell
        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]),
            receiver_xyz=np.array([10.0, 0.0, 0.0]),
            quantity="both",
        )

        assert result.source_xyz.shape == (3,)
        assert result.receiver_xyz.shape == (3,)
        assert result.sem_source_xyz.shape == (3,)
        assert result.time.shape == (50,)
        assert result.strain is not None
        assert result.strain.shape == (50, 6, 3)
        assert result.displacement is not None
        assert result.displacement.shape == (50, 3, 3)
        assert isinstance(result.n_tiles_used, int)
        assert isinstance(result.interpolation_used, bool)

    def test_query_quantity_strain_only(self, library_3d) -> None:
        """Query with quantity='strain' returns only strain."""
        lib = GreenFunctionLibrary(library_3d)

        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]),
            receiver_xyz=np.array([10.0, 0.0, 0.0]),
            quantity="strain",
        )

        assert result.strain is not None
        assert result.displacement is None

    def test_query_quantity_displacement_only(self, library_3d) -> None:
        """Query with quantity='displacement' returns only displacement."""
        lib = GreenFunctionLibrary(library_3d)

        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]),
            receiver_xyz=np.array([10.0, 0.0, 0.0]),
            quantity="displacement",
        )

        assert result.displacement is not None
        assert result.strain is None

    def test_receiver_nearest_match_selects_correct_source(self, library_3d) -> None:
        """Receiver nearest to source 0 selects source 0's run."""
        lib = GreenFunctionLibrary(library_3d)

        # receiver near (0, 0, 0) → nearest source at (0, 0, 0)
        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]),
            receiver_xyz=np.array([10.0, 0.0, 0.0]),
            quantity="strain",
        )
        assert np.allclose(result.sem_source_xyz, [0.0, 0.0, 0.0])

    def test_receiver_near_source_1_selects_source_1(self, library_3d) -> None:
        """Receiver near (1000, 0, 0) selects source at (1000, 0, 0)."""
        lib = GreenFunctionLibrary(library_3d)

        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]),
            receiver_xyz=np.array([990.0, 0.0, 0.0]),
            quantity="strain",
        )
        assert np.allclose(result.sem_source_xyz, [1000.0, 0.0, 0.0])

    def test_source_xyz_is_preserved(self, library_3d) -> None:
        """Result.source_xyz equals the query source_xyz."""
        lib = GreenFunctionLibrary(library_3d)

        source_pt = np.array([5.0, 5.0, 5.0])
        result = lib.query(
            source_xyz=source_pt, receiver_xyz=np.array([10.0, 0.0, 0.0]), quantity="strain"
        )
        assert np.allclose(result.source_xyz, source_pt)

    def test_receiver_xyz_is_preserved(self, library_3d) -> None:
        """Result.receiver_xyz equals the query receiver_xyz."""
        lib = GreenFunctionLibrary(library_3d)

        receiver_pt = np.array([10.0, 0.0, 0.0])
        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]), receiver_xyz=receiver_pt, quantity="strain"
        )
        assert np.allclose(result.receiver_xyz, receiver_pt)


# ---------------------------------------------------------------------------
# Tests — batch query
# ---------------------------------------------------------------------------


class TestGreenFunctionLibraryBatch:
    """GreenFunctionLibrary.query_batch() correctness."""

    def test_query_batch_returns_correct_number_of_results(self, library_3d) -> None:
        """Batch query returns one result per source."""
        lib = GreenFunctionLibrary(library_3d)

        sources = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        receivers = np.array([[10.0, 0.0, 0.0], [990.0, 0.0, 0.0]])

        results = lib.query_batch(sources, receivers, quantity="strain")
        assert len(results) == 2

        for r in results:
            assert r.strain is not None
            assert r.strain.shape == (50, 6, 3)

    def test_query_batch_single_receiver_broadcast(self, library_3d) -> None:
        """Single receiver broadcast to all sources."""
        lib = GreenFunctionLibrary(library_3d)

        sources = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        # Single receiver (3,) broadcast.
        receiver = np.array([10.0, 0.0, 0.0])

        results = lib.query_batch(sources, receiver, quantity="strain")
        assert len(results) == 2
        assert np.allclose(results[0].receiver_xyz, [10.0, 0.0, 0.0])

    def test_query_batch_single_receiver_2d_broadcast(self, library_3d) -> None:
        """Single receiver shape (1, 3) broadcast to all sources."""
        lib = GreenFunctionLibrary(library_3d)

        sources = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        receiver = np.array([[10.0, 0.0, 0.0]])

        results = lib.query_batch(sources, receiver, quantity="strain")
        assert len(results) == 2

    def test_query_batch_mismatch_raises(self, library_3d) -> None:
        """Mismatched source/receiver counts raise ValueError."""
        lib = GreenFunctionLibrary(library_3d)

        sources = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
        receivers = np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="Number of receivers"):
            lib.query_batch(sources, receivers)

    def test_query_batch_wrong_source_shape_raises(self, library_3d) -> None:
        """Wrong source shape raises ValueError."""
        lib = GreenFunctionLibrary(library_3d)

        with pytest.raises(ValueError, match="sources must have shape"):
            lib.query_batch(np.array([1.0, 2.0, 3.0]), np.array([[10.0, 0.0, 0.0]]))

    def test_query_batch_wrong_receiver_shape_raises(self, library_3d) -> None:
        """Wrong receiver shape raises ValueError."""
        lib = GreenFunctionLibrary(library_3d)

        with pytest.raises(ValueError, match="receivers must have shape"):
            lib.query_batch(np.array([[5.0, 5.0, 5.0]]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestGreenFunctionLibraryEdgeCases:
    """Edge cases for GreenFunctionLibrary."""

    def test_empty_library_raises_on_query(self, tmp_path: Path) -> None:
        """Query on an empty library raises ValueError."""
        lib = GreenFunctionLibrary(tmp_path)
        with pytest.raises(ValueError, match="No SEM sources available"):
            lib.query(source_xyz=np.array([0.0, 0.0, 0.0]), receiver_xyz=np.array([0.0, 0.0, 0.0]))

    def test_empty_library_raises_on_query_batch(self, tmp_path: Path) -> None:
        """Batch query on an empty library raises ValueError."""
        lib = GreenFunctionLibrary(tmp_path)
        with pytest.raises(ValueError, match="No SEM sources available"):
            lib.query_batch(
                sources=np.array([[0.0, 0.0, 0.0]]), receivers=np.array([[0.0, 0.0, 0.0]])
            )

    def test_source_xyz_at_receiver_returns_meaningful_data(self, library_3d) -> None:
        """Query with source_xyz == receiver_xyz location returns data.

        This exercises reciprocity: source_xyz is interpolated among the
        selected source run's vertices while receiver_xyz selects the run.
        """
        lib = GreenFunctionLibrary(library_3d)

        # Both coordinates near source 0's first tile.
        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]),
            receiver_xyz=np.array([5.0, 5.0, 5.0]),
            quantity="both",
        )

        assert result.strain is not None
        assert result.strain.shape == (50, 6, 3)
        assert not np.all(result.strain == 0.0), "Data should be non-zero"

    def test_source_xyz_at_sem_source_returns_data(self, library_3d) -> None:
        """Query with source_xyz at the SEM source itself returns data."""
        lib = GreenFunctionLibrary(library_3d)

        # receiver matches source 0, source_xyz = SEM source 0 location
        result = lib.query(
            source_xyz=np.array([0.0, 0.0, 0.0]),
            receiver_xyz=np.array([10.0, 0.0, 0.0]),
            quantity="strain",
        )

        assert result.strain is not None
        assert result.strain.shape == (50, 6, 3)

    def test_rebuild_index_flag_no_data_loss(self, library_3d) -> None:
        """rebuild_index=True returns correct data."""
        lib = GreenFunctionLibrary(library_3d, rebuild_index=True)

        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]),
            receiver_xyz=np.array([10.0, 0.0, 0.0]),
            quantity="strain",
        )

        assert result.strain is not None
        assert result.strain.shape == (50, 6, 3)

    def test_flat_single_source_directory_query(self, tmp_path: Path) -> None:
        """Flat postprocess output directory is readable as one source run."""
        rng = np.random.default_rng(7)
        time = np.linspace(0.0, 1.0, 50, dtype=np.float64)
        xs = np.array([0.0, 10.0])
        ys = np.array([0.0, 10.0])
        zs = np.array([0.0, 10.0])
        mesh = np.meshgrid(xs, ys, zs, indexing="ij")
        vertex_coords = np.column_stack([m.ravel() for m in mesh])
        vertex_ids = np.arange(vertex_coords.shape[0], dtype=np.int64)
        strain = rng.standard_normal((50, vertex_coords.shape[0], 6, 3)).astype(np.float32)
        displacement = rng.standard_normal((50, vertex_coords.shape[0], 3, 3)).astype(np.float32)

        _write_tile(
            tmp_path / "tile_x000_y000.h5",
            vertex_ids=vertex_ids,
            vertex_coords=vertex_coords,
            time=time,
            strain=strain,
            displacement=displacement,
            sem_source_xyz=np.array([100.0, 200.0, 0.0]),
        )

        lib = GreenFunctionLibrary(tmp_path)
        result = lib.query(
            source_xyz=np.array([5.0, 5.0, 5.0]),
            receiver_xyz=np.array([100.0, 200.0, 0.0]),
            quantity="both",
        )

        assert lib.n_sources == 1
        assert lib.n_tiles == 1
        assert np.allclose(result.sem_source_xyz, [100.0, 200.0, 0.0])
        assert result.strain is not None
        assert result.strain.shape == (50, 6, 3)
        assert result.displacement is not None
        assert result.displacement.shape == (50, 3, 3)

    def test_invalid_quantity_raises(self, library_3d) -> None:
        """Invalid quantity names fail loudly at the library API boundary."""
        lib = GreenFunctionLibrary(library_3d)
        with pytest.raises(ValueError, match="quantity"):
            lib.query(
                source_xyz=np.array([5.0, 5.0, 5.0]),
                receiver_xyz=np.array([10.0, 0.0, 0.0]),
                quantity="bogus_quantity",
            )
