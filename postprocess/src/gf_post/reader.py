"""HDF5 readers for checkpoint records and mesh geometry."""

import h5py
import numpy as np


class RecordReader:
    """Reads strain checkpoints from a single-rank record HDF5 file."""

    def __init__(self, path: str):
        self.path = path
        self._file: h5py.File | None = None

    def __enter__(self):
        self._file = h5py.File(self.path, "r")
        return self

    def __exit__(self, *args):
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def dt(self) -> float:
        return float(self._file.attrs["dt"])

    @property
    def source_direction(self) -> int:
        return int(self._file.attrs["source_direction"])

    @property
    def record_interval(self) -> int:
        return int(self._file.attrs["checkpoint_interval"])

    @property
    def nsteps(self) -> int:
        return int(self._file.attrs["nsteps"])

    @property
    def local_element_ids(self) -> np.ndarray:
        return np.array(self._file["local_element_ids"])

    @property
    def n_records(self) -> int:
        strain = self._file["strain"]
        return strain.shape[0]

    def read_strain(self, step: int) -> np.ndarray:
        """Read strain for a single timestep.

        Returns shape: (n_elem_local, NGLL, NGLL, NGLL, 6)
        """
        return np.array(self._file["strain"][step])

    def read_all_strain(self) -> np.ndarray:
        """Read all strain checkpoints.

        Returns shape: (n_records, n_elem_local, NGLL, NGLL, NGLL, 6)
        """
        return np.array(self._file["strain"])


class GeometryReader:
    """Reads GLL geometry data from mesh.h5."""

    def __init__(self, path: str):
        self.path = path
        self._file: h5py.File | None = None

    def __enter__(self):
        self._file = h5py.File(self.path, "r")
        return self

    def __exit__(self, *args):
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def coords(self) -> np.ndarray:
        """GLL node coordinates [n_cell, NGLL, NGLL, NGLL, 3]."""
        return np.array(self._file["/field/element/coords"])

    @property
    def dxi_dx(self) -> np.ndarray:
        """Jacobian inverse [n_cell, NGLL, NGLL, NGLL, 3, 3]."""
        return np.array(self._file["/field/element/dxi_dx"])

    @property
    def is_pml(self) -> np.ndarray:
        """PML flag per element [n_cell]."""
        return np.array(self._file["/field/element/is_pml"])

    @property
    def n_cell(self) -> int:
        return self.coords.shape[0]

    @property
    def ngll(self) -> int:
        return self.coords.shape[1]


def merge_records(rank_files: list[str], n_cell: int) -> tuple[np.ndarray, dict]:
    """Merge strain from multiple rank record files into unified view.

    Args:
        rank_files: list of paths to record_{r}.h5 files.
        n_cell: total number of elements (global).

    Returns:
        (merged_strain, info) where merged_strain is
        [n_records, n_cell, NGLL, NGLL, NGLL, 6] and info contains metadata.
    """
    # Open all files to get shapes
    readers = [RecordReader(path) for path in rank_files]
    for r in readers:
        r.__enter__()

    n_records = readers[0].n_records
    ngll = readers[0].read_strain(0).shape[1]

    merged = np.zeros(
        (n_records, n_cell, ngll, ngll, ngll, 6), dtype=readers[0].read_strain(0).dtype
    )

    for r in readers:
        eids = r.local_element_ids  # 1-based global IDs
        strain = r.read_all_strain()
        for i, eid in enumerate(eids):
            merged[:, eid - 1, :, :, :, :] = strain[:, i, :, :, :, :]

    info = {
        "dt": readers[0].dt,
        "nsteps": readers[0].nsteps,
        "checkpoint_interval": readers[0].record_interval,
    }

    for r in readers:
        r.__exit__(None, None, None)

    return merged, info
