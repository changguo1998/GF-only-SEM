"""HDF5 writer for Green's function tile files.

Writes the 3×6 strain Green's tensor at all GLL nodes, spatially tiled
by element range. No receiver positions — output is the full field.
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
import h5py


STRAIN_NAMES = ["strain_xx", "strain_yy", "strain_zz",
                "strain_xy", "strain_xz", "strain_yz"]
FORCE_DIRECTIONS = ["fx", "fy", "fz"]


class GFWriter:
    """Writes strain Green's functions as spatially-tiled HDF5 files."""
    
    @staticmethod
    def write(
        output_dir: str | Path,
        gll_coords: npt.NDArray[np.float64],
        time: npt.NDArray[np.float64],
        dt: float,
        greens: npt.NDArray[np.float64],
        tile_elems: int = 100,
    ) -> List[Path]:
        """Write Green's function output as spatially-tiled HDF5 files.
        
        Args:
            output_dir: Directory to write tile_{i}.h5 files.
            gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] GLL node coordinates.
            time: [nt] time array.
            dt: Time step.
            greens: [nt, n_cell, NGLL, NGLL, NGLL, 6, 3] Green's tensor.
            tile_elems: Max elements per tile.
        
        Returns:
            List of written tile file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_cell = gll_coords.shape[0]
        nt = len(time)
        _, _, ngll, _, _, n_comp, n_dir = greens.shape
        
        tiles = []
        for start in range(0, n_cell, tile_elems):
            end = min(start + tile_elems, n_cell)
            tile_idx = start // tile_elems
            tile_path = output_dir / f"tile_{tile_idx}.h5"
            
            _write_tile(tile_path, gll_coords[start:end], time, dt,
                        greens[:, start:end], tile_idx, start)
            tiles.append(tile_path)
        
        return tiles


def _write_tile(
    path: Path,
    tile_coords: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    dt: float,
    tile_greens: npt.NDArray[np.float64],
    tile_idx: int,
    global_elem_start: int,
):
    """Write a single tile HDF5 file."""
    n_local_cell = tile_coords.shape[0]
    nt = len(time)
    global_elem_end = global_elem_start + n_local_cell - 1
    
    with h5py.File(path, "w") as f:
        f.attrs["description"] = (
            f"Strain Green's functions, tile {tile_idx} "
            f"(elements {global_elem_start}-{global_elem_end})"
        )
        f.attrs["version"] = "0.2.0"
        f.attrs["ncell"] = n_local_cell
        f.attrs["ngll"] = tile_coords.shape[1]
        f.attrs["elem_start"] = global_elem_start
        
        # /time/
        time_grp = f.create_group("time")
        time_grp.create_dataset("t", data=time.astype(np.float64))
        time_grp.attrs["dt"] = np.float64(dt)
        time_grp.attrs["nsteps"] = nt
        
        # /field/ — Green's tensor at GLL nodes
        field = f.create_group("field")
        field.create_dataset("greens_tensor",
                             data=tile_greens.astype(np.float32))
        field.create_dataset("coords",
                             data=tile_coords.astype(np.float32))