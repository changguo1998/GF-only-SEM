"""Shared test fixtures for the greenfun module."""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


@dataclasses.dataclass
class GreenfunTestLibrary:
    """A synthetic greenfun library for testing.

    Creates a temporary directory with one or more SEM-source directories,
    each containing tile HDF5 files with realistic but random data.
    """

    root: Path
    n_sources: int
    n_tiles_per_source: int
    n_vertex_per_tile: int
    nt: int
    tile_x_size: float = 100.0
    tile_y_size: float = 100.0

    @property
    def n_vertex_total(self) -> int:
        """Total number of unique vertices (may have overlap)."""
        return self.n_tiles_per_source * self.n_vertex_per_tile

    @staticmethod
    def create(
        n_sources: int = 3,
        n_tiles_per_source: int = 2,
        n_vertex_per_tile: int = 8,
        nt: int = 100,
        include_displacement: bool = True,
    ) -> GreenfunTestLibrary:
        """Create a synthetic greenfun library and return the handle.

        The caller is responsible for calling ``cleanup()``.
        """
        root = Path(tempfile.mkdtemp(prefix="greenfun_test_"))
        obj = GreenfunTestLibrary(
            root=root,
            n_sources=n_sources,
            n_tiles_per_source=n_tiles_per_source,
            n_vertex_per_tile=n_vertex_per_tile,
            nt=nt,
        )
        obj._populate(include_displacement)
        return obj

    def _populate(self, include_displacement: bool) -> None:
        """Populate the library directory with synthetic tile data."""
        rng = np.random.default_rng(42)
        time = np.linspace(0.0, 1.0, self.nt, dtype=np.float64)

        # Place SEM sources along a line / grid.
        sem_sources = np.column_stack([
            np.linspace(0.0, 1000.0, self.n_sources),
            np.zeros(self.n_sources),
            np.zeros(self.n_sources),
        ])

        for src_idx in range(self.n_sources):
            src_dir = self.root / f"src_{src_idx:04d}"
            src_dir.mkdir(parents=True)
            sem_xyz = sem_sources[src_idx]

            for tile_idx in range(self.n_tiles_per_source):
                tile_path = src_dir / f"tile_x{tile_idx:03d}_y{0:03d}.h5"
                n_vert = self.n_vertex_per_tile

                # Generate a small grid of vertex coordinates at z=0.
                gx, gy = np.meshgrid(
                    np.linspace(tile_idx * self.tile_x_size, (tile_idx + 1) * self.tile_x_size, int(np.sqrt(n_vert))),
                    np.linspace(0, self.tile_y_size, int(np.sqrt(n_vert))),
                )
                gx = gx.ravel()[:n_vert]
                gy = gy.ravel()[:n_vert]
                vertex_coords = np.column_stack([gx, gy, np.zeros(n_vert)])

                vertex_ids = np.arange(
                    tile_idx * n_vert, tile_idx * n_vert + n_vert, dtype=np.int64
                )

                with h5py.File(tile_path, "w") as h5:
                    # Attrs
                    h5.attrs["version"] = "1.0.0"
                    h5.attrs["basis"] = "mesh_vertices"
                    h5.attrs["tile_x_index"] = tile_idx
                    h5.attrs["tile_y_index"] = 0
                    h5.attrs["x_min_m"] = float(vertex_coords[:, 0].min())
                    h5.attrs["x_max_m"] = float(vertex_coords[:, 0].max())
                    h5.attrs["y_min_m"] = float(vertex_coords[:, 1].min())
                    h5.attrs["y_max_m"] = float(vertex_coords[:, 1].max())
                    h5.attrs["z_min_m"] = float(vertex_coords[:, 2].min())
                    h5.attrs["z_max_m"] = float(vertex_coords[:, 2].max())
                    h5.attrs["record_depth_max_m"] = 0.0
                    h5.attrs["record_depth_actual_m"] = 0.0
                    h5.attrs["excludes_pml"] = 1
                    h5.attrs["source_xyz_m"] = sem_xyz
                    h5.attrs["source_directions"] = "x,y,z"
                    if include_displacement:
                        h5.attrs["greens_quantities"] = "strain,displacement"
                    else:
                        h5.attrs["greens_quantities"] = "strain"

                    # Time
                    h5.create_dataset("/time/t", data=time)

                    # Mesh
                    h5.create_dataset("/mesh/vertex_ids", data=vertex_ids)
                    h5.create_dataset(
                        "/mesh/vertex_coords", data=vertex_coords, dtype=np.float64
                    )

                    # Strain Green tensor [nt, n_vert, 6, 3]
                    strain_data = rng.standard_normal(
                        (self.nt, n_vert, 6, 3), dtype=np.float32
                    )
                    h5.create_dataset(
                        "/field/greens_tensor",
                        data=strain_data,
                        dtype=np.float32,
                        compression="gzip",
                        compression_opts=4,
                        shuffle=True,
                    )

                    # Displacement tensor [nt, n_vert, 3, 3]
                    if include_displacement:
                        disp_data = rng.standard_normal(
                            (self.nt, n_vert, 3, 3), dtype=np.float32
                        )
                        h5.create_dataset(
                            "/field/displacement_tensor",
                            data=disp_data,
                            dtype=np.float32,
                            compression="gzip",
                            compression_opts=4,
                            shuffle=True,
                        )

    def cleanup(self) -> None:
        """Remove the temporary library directory."""
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def greenfun_library(request: pytest.FixtureRequest) -> GreenfunTestLibrary:
    """Fixture that creates a synthetic greenfun library for testing.

    Cleanup is automatic after the test.
    """
    lib = GreenfunTestLibrary.create(
        n_sources=3,
        n_tiles_per_source=2,
        n_vertex_per_tile=9,
        nt=100,
        include_displacement=True,
    )
    request.addfinalizer(lib.cleanup)
    return lib


@pytest.fixture
def greens_tile_factory(tmp_path: Path):
    """Fixture that returns a callable to create ad-hoc tile files.

    The returned function takes (n_vertices, nt, include_displacement)
    and writes a single tile HDF5, returning the path.
    """

    def _make(
        n_vertices: int = 8,
        nt: int = 50,
        include_displacement: bool = True,
    ) -> Path:
        rng = np.random.default_rng(12345)
        tile = tmp_path / "tile_x000_y000.h5"
        time = np.linspace(0.0, 0.5, nt, dtype=np.float64)
        coords = rng.uniform(0, 100, (n_vertices, 3)).astype(np.float64)
        with h5py.File(tile, "w") as h5:
            h5.attrs["version"] = "1.0.0"
            h5.attrs["basis"] = "mesh_vertices"
            h5.attrs["tile_x_index"] = 0
            h5.attrs["tile_y_index"] = 0
            h5.attrs["x_min_m"] = float(coords[:, 0].min())
            h5.attrs["x_max_m"] = float(coords[:, 0].max())
            h5.attrs["y_min_m"] = float(coords[:, 1].min())
            h5.attrs["y_max_m"] = float(coords[:, 1].max())
            h5.attrs["z_min_m"] = float(coords[:, 2].min())
            h5.attrs["z_max_m"] = float(coords[:, 2].max())
            h5.attrs["record_depth_max_m"] = 0.0
            h5.attrs["record_depth_actual_m"] = 0.0
            h5.attrs["excludes_pml"] = 1
            h5.attrs["source_xyz_m"] = np.array([500.0, 0.0, 0.0])
            h5.attrs["source_directions"] = "x,y,z"
            h5.attrs["greens_quantities"] = "strain,displacement" if include_displacement else "strain"
            h5.create_dataset("/time/t", data=time)
            h5.create_dataset("/mesh/vertex_ids", data=np.arange(n_vertices, dtype=np.int64))
            h5.create_dataset("/mesh/vertex_coords", data=coords)
            h5.create_dataset(
                "/field/greens_tensor",
                data=rng.standard_normal((nt, n_vertices, 6, 3)).astype(np.float32),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            if include_displacement:
                h5.create_dataset(
                    "/field/displacement_tensor",
                    data=rng.standard_normal((nt, n_vertices, 3, 3)).astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
        return tile

    return _make