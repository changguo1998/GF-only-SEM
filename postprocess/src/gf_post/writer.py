"""HDF5 writer for Green's function tile files."""

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt
import h5py


class GFWriter:
    """Writes strain Green's functions as spatially-tiled HDF5 files."""
    
    @staticmethod
    def write(
        output_dir: str | Path,
        receivers: npt.NDArray[np.float64],
        time: npt.NDArray[np.float64],
        waveforms: dict[str, npt.NDArray[np.float64]],
        tile_size: int = 1000,
    ) -> list[Path]:
        """Write Green's function output as spatially-tiled HDF5 files.
        
        Args:
            output_dir: directory to write tile_{i}.h5 files.
            receivers: [n_recv, 3] receiver positions.
            time: [nt] time array.
            waveforms: dict {"fx": ..., "fy": ..., "fz": ...}, each [nt, n_recv, 6].
            tile_size: max receivers per tile.
        
        Returns:
            list of written tile file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        n_recv = len(receivers)
        nt = len(time)
        dt = float(time[1] - time[0]) if nt > 1 else 0.0
        
        # Simple tiling: split receivers into contiguous tiles by index
        tiles = []
        for start in range(0, n_recv, tile_size):
            end = min(start + tile_size, n_recv)
            tile_receivers = receivers[start:end]
            tile_idx = start // tile_size
            
            tile_path = output_dir / f"tile_{tile_idx}.h5"
            _write_tile(tile_path, tile_receivers, time, dt, tile_size, waveforms, start)
            tiles.append(tile_path)
        
        return tiles


def _write_tile(
    path: Path,
    tile_receivers: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    dt: float,
    tile_size: int,
    waveforms: dict[str, npt.NDArray[np.float64]],
    global_start: int,
):
    """Write a single tile HDF5 file."""
    n_recv = len(tile_receivers)
    nt = len(time)
    
    with h5py.File(path, "w") as f:
        # File-level attributes
        f.attrs["description"] = f"Strain Green's functions, tile {global_start // tile_size}"
        f.attrs["version"] = "0.1.0"
        f.attrs["nreceivers"] = n_recv
        
        # Spatial extent
        f.attrs["minx"] = float(tile_receivers[:, 0].min())
        f.attrs["maxx"] = float(tile_receivers[:, 0].max())
        f.attrs["miny"] = float(tile_receivers[:, 1].min())
        f.attrs["maxy"] = float(tile_receivers[:, 1].max())
        f.attrs["minz"] = float(tile_receivers[:, 2].min())
        f.attrs["maxz"] = float(tile_receivers[:, 2].max())
        
        # /receivers/
        recv = f.create_group("receivers")
        recv.create_dataset("positions", data=tile_receivers.astype(np.float64))
        recv.create_dataset("names", data=np.array(
            [f"recv_{global_start + i:04d}" for i in range(n_recv)], dtype="S32"
        ))
        
        # /time/
        time_grp = f.create_group("time")
        time_grp.create_dataset("t", data=time.astype(np.float64))
        time_grp.create_dataset("dt", data=np.float64(dt))
        
        # /waveforms/
        wf = f.create_group("waveforms")
        for direction in ["fx", "fy", "fz"]:
            wf_dir = wf.create_group(direction)
            strain_data = waveforms[direction][:, global_start:global_start + n_recv, :]
            for i in range(n_recv):
                recv_grp = wf_dir.create_group(f"recv_{global_start + i:04d}")
                for c, name in enumerate(["strain_xx", "strain_yy", "strain_zz",
                                           "strain_xy", "strain_xz", "strain_yz"]):
                    recv_grp.create_dataset(name, data=strain_data[:, i, c].astype(np.float64))


