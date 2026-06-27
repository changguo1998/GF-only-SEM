"""HDF5 writer for spatially-tiled Green's function output.

Output tiles cover xy-spatial bins of size green_tile_size_m.
Each tile contains vertex_ids + greens_tensor for all vertices in that bin.
Consumers read coordinates from mesh.h5 via vertex_ids.
"""

from pathlib import Path
from typing import List

import h5py
import numpy as np
import numpy.typing as npt


class GFWriter:
    """Writes strain Green's functions as spatially-tiled HDF5 files."""

    @staticmethod
    def write(
        output_dir: str | Path,
        vertex_coords: npt.NDArray[np.float64],
        vertex_ids: npt.NDArray[np.int64],
        time: npt.NDArray[np.float64],
        dt: float,
        greens: npt.NDArray[np.float64],
        green_tile_size_m: float,
        domain_bounds: dict[str, float],
        record_depth_max_m: float = 0.0,
        record_depth_actual_m: float = 0.0,
    ) -> List[Path]:
        """Write Green's function output as spatially-tiled HDF5 files.

        Args:
            output_dir: Directory to write tile_x{i}_y{j}.h5 files.
            vertex_coords: [n_vertex, 3] mesh vertex coordinates.
            vertex_ids: [n_vertex] global mesh vertex IDs (1-based).
            time: [nt] time array.
            dt: Time step.
            greens: [nt, n_vertex, 6, 3] Green's tensor at vertices only.
            green_tile_size_m: Horizontal tile width in meters.
            domain_bounds: dict with xmin, xmax, ymin, ymax, zmin, zmax.
            record_depth_max_m: Max recording depth.
            record_depth_actual_m: Snapped actual recording depth.

        Returns:
            List of written tile file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_vertex = vertex_coords.shape[0]
        nt = len(time)
        _, _, n_comp, n_dir = greens.shape

        xmin = domain_bounds["xmin"]
        ymin = domain_bounds["ymin"]
        zmin = domain_bounds["zmin"]
        zmax = domain_bounds["zmax"]
        xmax = domain_bounds["xmax"]
        ymax = domain_bounds["ymax"]

        # Bin vertices into spatial tiles
        tile_bins: dict[tuple[int, int], list[int]] = {}
        for vi in range(n_vertex):
            x, y = vertex_coords[vi, 0], vertex_coords[vi, 1]
            tx = int(np.floor((x - xmin) / green_tile_size_m))
            ty = int(np.floor((y - ymin) / green_tile_size_m))
            key = (tx, ty)
            if key not in tile_bins:
                tile_bins[key] = []
            tile_bins[key].append(vi)

        tiles = []
        for (tx, ty), vert_indices in sorted(tile_bins.items()):
            if not vert_indices:
                continue

            tile_path = output_dir / f"tile_x{tx:03d}_y{ty:03d}.h5"
            _write_tile(
                tile_path,
                vertex_ids[vert_indices],
                time, dt,
                greens[:, vert_indices, :, :],
                tx, ty,
                xmin + tx * green_tile_size_m,
                xmin + (tx + 1) * green_tile_size_m,
                ymin + ty * green_tile_size_m,
                ymin + (ty + 1) * green_tile_size_m,
                zmin, zmax,
                record_depth_max_m, record_depth_actual_m,
            )
            tiles.append(tile_path)

        return tiles


def _write_tile(
    path: Path,
    tile_vertex_ids: npt.NDArray[np.int64],
    time: npt.NDArray[np.float64],
    dt: float,
    tile_greens: npt.NDArray[np.float64],
    tile_x: int,
    tile_y: int,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    y_max_m: float,
    z_min_m: float,
    z_max_m: float,
    record_depth_max_m: float,
    record_depth_actual_m: float,
):
    """Write a single tile HDF5 file."""
    nt = len(time)
    n_tile_vertices = tile_vertex_ids.shape[0]

    with h5py.File(path, "w") as f:
        # Attrs
        f.attrs["version"] = "1.0.0"
        f.attrs["basis"] = "mesh_vertices"
        f.attrs["tile_x_index"] = tile_x
        f.attrs["tile_y_index"] = tile_y
        f.attrs["x_min_m"] = x_min_m
        f.attrs["x_max_m"] = x_max_m
        f.attrs["y_min_m"] = y_min_m
        f.attrs["y_max_m"] = y_max_m
        f.attrs["z_min_m"] = z_min_m
        f.attrs["z_max_m"] = z_max_m
        f.attrs["record_depth_max_m"] = record_depth_max_m
        f.attrs["record_depth_actual_m"] = record_depth_actual_m
        f.attrs["excludes_pml"] = True

        # Time
        time_grp = f.create_group("time")
        time_grp.create_dataset("t", data=time.astype(np.float64))
        time_grp.attrs["dt"] = np.float64(dt)
        time_grp.attrs["nsteps"] = nt

        # Mesh
        mesh_grp = f.create_group("mesh")
        mesh_grp.create_dataset("vertex_ids", data=tile_vertex_ids.astype(np.int64))

        # Field
        field_grp = f.create_group("field")
        field_grp.create_dataset(
            "greens_tensor", data=tile_greens.astype(np.float32),
        )