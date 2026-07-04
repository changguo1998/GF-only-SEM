"""HDF5 writer for element-tiled or spatially-tiled Green's function output.

Output tiles partition by element count (default) or spatial bin size
(when green_tile_size_m is set).
Each tile contains vertex_ids + greens_tensor for all vertices in that bin.
Consumers read coordinates from model.h5 via vertex_ids.
"""

from pathlib import Path
from typing import List

import h5py
import numpy as np
import numpy.typing as npt


def _find_tile_index(interior_idx: int, tile_sizes: list[int]) -> int:
    """Return tile index for an interior element index.

    Given tile_sizes like [5, 5] (two tiles of 5 elements each),
    interior element index 0-4 → tile 0, 5-9 → tile 1.
    """
    cum = 0
    for t, sz in enumerate(tile_sizes):
        cum += sz
        if interior_idx < cum:
            return t
    return len(tile_sizes) - 1


class GFWriter:
    """Writes strain Green's functions as element-tiled or spatially-tiled HDF5 files."""

    @staticmethod
    def write(
        output_dir: str | Path,
        vertex_coords: npt.NDArray[np.float64],
        vertex_ids: npt.NDArray[np.int64],
        time: npt.NDArray[np.float64],
        solver_dt_s: float,
        greens: npt.NDArray[np.float64],
        nx_elements: int,
        ny_elements: int,
        pml_thickness: dict[str, int],
        tilex_elements: list[int],
        tiley_elements: list[int],
        domain_bounds: dict[str, float],
        green_tile_size_m: float | None = None,
        record_depth_max_m: float = 0.0,
        record_depth_actual_m: float = 0.0,
    ) -> List[Path]:
        """Write Green's function output as tiled HDF5 files.

        Tiles partition by element count (default) or spatial bin size
        (when green_tile_size_m is set).

        Args:
            output_dir: Directory to write tile_x{i}_y{j}.h5 files.
            vertex_coords: [n_vertex, 3] mesh vertex coordinates.
            vertex_ids: [n_vertex] global mesh vertex IDs (1-based).
            time: [nt] time array.
            solver_dt_s: Solver timestep in seconds.
            greens: [nt, n_vertex, 6, 3] Green's tensor at vertices only.
            nx_elements: Total elements in x.
            ny_elements: Total elements in y.
            pml_thickness: Dict with xmin, xmax, ymin, ymax, zmin, zmax.
            tilex_elements: List of tile sizes (in elements) along x.
            tiley_elements: List of tile sizes (in elements) along y.
            domain_bounds: dict with xmin, xmax, ymin, ymax, zmin, zmax.
            green_tile_size_m: Spatial tile size in meters. When set, uses
                spatial binning instead of element-count tiling.
            record_depth_max_m: Max recording depth.
            record_depth_actual_m: Snapped actual recording depth.

        Returns:
            List of written tile file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_vertex = vertex_coords.shape[0]
        nt = len(time)

        xmin = domain_bounds["xmin"]
        ymin = domain_bounds["ymin"]
        zmin = domain_bounds["zmin"]
        zmax = domain_bounds["zmax"]
        xmax = domain_bounds["xmax"]
        ymax = domain_bounds["ymax"]

        # Validate input shapes
        expected_shape = (len(time), vertex_coords.shape[0], 6, 3)
        if greens.shape != expected_shape:
            raise ValueError(
                f"greens shape mismatch: got {greens.shape}, expected {expected_shape}"
            )

        # Choose tiling method: spatial or element-count
        tile_bins: dict[tuple[int, int], list[int]] = {}
        use_spatial = green_tile_size_m is not None and green_tile_size_m > 0

        if use_spatial:
            # Spatial binning — vertices grouped by coordinate bins
            for vi in range(n_vertex):
                x, y = vertex_coords[vi, 0], vertex_coords[vi, 1]
                tx = int(np.floor((x - xmin) / green_tile_size_m))
                ty = int(np.floor((y - ymin) / green_tile_size_m))
                key = (tx, ty)
                if key not in tile_bins:
                    tile_bins[key] = []
                tile_bins[key].append(vi)
        else:
            # Element-count binning (original logic)
            dx = (xmax - xmin) / nx_elements if nx_elements > 0 else 0.0
            dy = (ymax - ymin) / ny_elements if ny_elements > 0 else 0.0

            pml_xmin = pml_thickness.get("xmin", 0)
            pml_xmax = pml_thickness.get("xmax", 0)
            pml_ymin = pml_thickness.get("ymin", 0)
            pml_ymax = pml_thickness.get("ymax", 0)

            tile_x_cum = [0]
            for sz in tilex_elements:
                tile_x_cum.append(tile_x_cum[-1] + sz)
            tile_y_cum = [0]
            for sz in tiley_elements:
                tile_y_cum.append(tile_y_cum[-1] + sz)

            for vi in range(n_vertex):
                x, y = vertex_coords[vi, 0], vertex_coords[vi, 1]

                ei = int(np.floor((x - xmin) / dx)) if dx > 0 else 0
                ej = int(np.floor((y - ymin) / dy)) if dy > 0 else 0
                ei = min(ei, nx_elements - 1) if nx_elements > 0 else 0
                ej = min(ej, ny_elements - 1) if ny_elements > 0 else 0

                interior_i = ei - pml_xmin
                interior_j = ej - pml_ymin

                if interior_i < 0 or interior_i >= tile_x_cum[-1]:
                    continue
                if interior_j < 0 or interior_j >= tile_y_cum[-1]:
                    continue

                tx = _find_tile_index(interior_i, tilex_elements)
                ty = _find_tile_index(interior_j, tiley_elements)
                key = (tx, ty)
                if key not in tile_bins:
                    tile_bins[key] = []
                tile_bins[key].append(vi)
        tiles = []
        for (tx, ty), vert_indices in sorted(tile_bins.items()):
            if not vert_indices:
                continue

            # Compute physical bounds of this tile
            if use_spatial:
                tile_x_min = xmin + tx * green_tile_size_m
                tile_x_max = xmin + (tx + 1) * green_tile_size_m
                tile_y_min = ymin + ty * green_tile_size_m
                tile_y_max = ymin + (ty + 1) * green_tile_size_m
            else:
                i_start = pml_xmin + tile_x_cum[tx]
                i_end = pml_xmin + tile_x_cum[tx + 1]
                j_start = pml_ymin + tile_y_cum[ty]
                j_end = pml_ymin + tile_y_cum[ty + 1]

                tile_x_min = xmin + i_start * dx
                tile_x_max = xmin + i_end * dx
                tile_y_min = ymin + j_start * dy
                tile_y_max = ymin + j_end * dy

            tile_path = output_dir / f"tile_x{tx:03d}_y{ty:03d}.h5"
            _write_tile(
                tile_path,
                vertex_ids[vert_indices],
                time,
                solver_dt_s,
                greens[:, vert_indices, :, :],
                tx,
                ty,
                tile_x_min,
                tile_x_max,
                tile_y_min,
                tile_y_max,
                zmin,
                zmax,
                record_depth_max_m,
                record_depth_actual_m,
            )
            tiles.append(tile_path)

        return tiles


def _write_tile(
    path: Path,
    tile_vertex_ids: npt.NDArray[np.int64],
    time: npt.NDArray[np.float64],
    solver_dt_s: float,
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
        time_grp.attrs["dt"] = np.float64(solver_dt_s)
        time_grp.attrs["nsteps"] = len(time)

        # Mesh
        mesh_grp = f.create_group("mesh")
        mesh_grp.create_dataset("vertex_ids", data=tile_vertex_ids.astype(np.int64))

        # Field
        field_grp = f.create_group("field")
        field_grp.create_dataset("greens_tensor", data=tile_greens.astype(np.float32))
