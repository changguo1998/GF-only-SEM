"""HDF5 readers for record files and mesh geometry.

Record files now store shallow mesh-vertex strain (from recording map)
with vertex_ids. Metadata (solver_dt, nsteps, tilex_elements, tiley_elements) comes from config.h5.
"""

import sys

import h5py
import numpy as np
import numpy.typing as npt


class RecordReader:
    """Reads shallow mesh-vertex strain from a single-rank record HDF5 file.

    File format (wavefields/{direction}/record_{r}.h5):
      attrs: rank, source_direction, basis="mesh_vertices", excludes_pml
      /vertex_ids  : int64[n_vertices]
      /strain      : float32[n_snapshots, n_vertices, 6]  (extendible)
    """

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
    def source_direction(self) -> str:
        return str(self._file.attrs["source_direction"])

    @property
    def basis(self) -> str:
        return str(self._file.attrs.get("basis", "mesh_vertices"))

    @property
    def vertex_ids(self) -> np.ndarray:
        """Global mesh vertex IDs recorded by this rank [n_vertices]."""
        return np.array(self._file["vertex_ids"])

    @property
    def n_vertices(self) -> int:
        return int(self._file["vertex_ids"].shape[0])

    @property
    def n_snapshots(self) -> int:
        return int(self._file["strain"].shape[0])

    def read_strain(self, snap_idx: int) -> np.ndarray:
        """Read strain for one snapshot.

        Returns shape: (n_vertices, 6)
        """
        return np.array(self._file["strain"][snap_idx])

    def read_all_strain(self) -> np.ndarray:
        """Read all strain snapshots.

        Returns shape: (n_snapshots, n_vertices, 6)
        """
        return np.array(self._file["strain"])


class GeometryReader:
    """Reads mesh vertex coordinates from model.h5."""

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
    def vertex_coords(self) -> np.ndarray:
        """Mesh vertex coordinates [n_vertex, 3]."""
        return np.array(self._file["/topology/vertex_to_coord"])

    @property
    def n_vertex(self) -> int:
        return self.vertex_coords.shape[0]

    @property
    def domain_bounds(self) -> dict[str, float]:
        """Domain bounds from /domain/ attrs."""
        domain = self._file["/domain"]
        return {
            "xmin": float(domain.attrs["xmin"]),
            "xmax": float(domain.attrs["xmax"]),
            "ymin": float(domain.attrs["ymin"]),
            "ymax": float(domain.attrs["ymax"]),
            "zmin": float(domain.attrs["zmin"]),
            "zmax": float(domain.attrs["zmax"]),
        }


class ConfigReader:
    """Reads simulation config from config.h5."""

    def __init__(self, path: str):
        self._file = h5py.File(path, "r")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._file.close()

    @property
    def nx_elements(self) -> int:
        return int(self._file["/simulation"].attrs.get("nx_elements", 0))

    @property
    def ny_elements(self) -> int:
        return int(self._file["/simulation"].attrs.get("ny_elements", 0))

    @property
    def pml_thickness(self) -> dict[str, int]:
        return {
            "xmin": int(self._file["/simulation"].attrs.get("pml_xmin", 0)),
            "xmax": int(self._file["/simulation"].attrs.get("pml_xmax", 0)),
            "ymin": int(self._file["/simulation"].attrs.get("pml_ymin", 0)),
            "ymax": int(self._file["/simulation"].attrs.get("pml_ymax", 0)),
            "zmin": int(self._file["/simulation"].attrs.get("pml_zmin", 0)),
            "zmax": int(self._file["/simulation"].attrs.get("pml_zmax", 0)),
        }

    @property
    def tilex_elements(self) -> list[int]:
        return list(self._file["/simulation/tilex_elements"][:])

    @property
    def tiley_elements(self) -> list[int]:
        return list(self._file["/simulation/tiley_elements"][:])

    @property
    def record_depth_max_m(self) -> float:
        return float(self._file["/simulation"].attrs.get("record_depth_max_m", 0.0))

    @property
    def record_depth_actual_m(self) -> float:
        return float(self._file["/simulation"].attrs.get("record_depth_actual_m", 0.0))

    @property
    def solver_dt(self) -> float:
        return float(self._file["/simulation"].attrs.get("solver_dt", 0.01))

    @property
    def nsteps(self) -> int:
        return int(self._file["/simulation"].attrs.get("nsteps", 0))

    @property
    def output_dt_s(self) -> float:
        return float(self._file["/simulation"].attrs.get("output_dt_s", self.solver_dt))


def merge_vertex_records(rank_files: list[str], n_vertex: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertex-level strain from multiple rank record files.

    Each rank recorded a subset of global mesh vertices. This function
    assembles the full-mesh strain array.

    Args:
        rank_files: list of paths to record_{r}.h5 files.
        n_vertex: total number of unique mesh vertices (global).

    Returns:
        (merged_strain, vertex_mask) where
          merged_strain: [n_snapshots, n_vertex, 6] float32
          vertex_mask:   [n_vertex] bool — True for vertices that were recorded
    """
    readers = [RecordReader(p) for p in rank_files]
    for rank_reader in readers:
        rank_reader.__enter__()

    try:
        n_snapshots = readers[0].n_snapshots
        dtype = readers[0].read_strain(0).dtype

        merged = np.zeros((n_snapshots, n_vertex, 6), dtype=dtype)
        mask = np.zeros(n_vertex, dtype=bool)

        for rank_reader in readers:
            local_vertex_ids = rank_reader.vertex_ids  # 1-based global vertex IDs
            strain = rank_reader.read_all_strain()  # [n_snapshots, n_local_vertices, 6]
            for local_index, global_vertex_id in enumerate(local_vertex_ids):
                zero_based_index = int(global_vertex_id) - 1
                if 0 <= zero_based_index < n_vertex:
                    if mask[zero_based_index]:
                        print(
                            f"[postprocess] WARNING: vertex {global_vertex_id} "
                            f"recorded by multiple ranks — using last value",
                            file=sys.stderr,
                        )
                    merged[:, zero_based_index, :] = strain[:, local_index, :]
                    mask[zero_based_index] = True
    finally:
        for rank_reader in readers:
            rank_reader.__exit__(None, None, None)

    return merged, mask
