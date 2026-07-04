"""HDF5 readers for per-step record files and mesh geometry.

Record files are now per-step: wavefields/{direction}/record_{rank}_{step}.h5,
each containing a single snapshot (shape [1, n_vertices, 6]).
Metadata (solver_dt, nsteps, tilex_elements, tiley_elements) comes from config.h5.
"""

import sys

import h5py
import numpy as np
import numpy.typing as npt


class RecordReader:
    """Reads a single-step record HDF5 file.

    File format (wavefields/{direction}/record_{r}_{step}.h5):
      attrs: rank, source_direction, basis="mesh_vertices", excludes_pml
      /vertex_ids  : int64[n_vertices]
      /strain      : float32[1, n_vertices, 6]  (single snapshot)
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
        """Number of snapshots in this file (always 1 for per-step files)."""
        return int(self._file["strain"].shape[0])

    def read_strain(self, snap_idx: int = 0) -> np.ndarray:
        """Read strain for one snapshot.

        Args:
            snap_idx: Must be 0 for per-step files.

        Returns shape: (n_vertices, 6)
        """
        return np.array(self._file["strain"][snap_idx])

    def read_all_strain(self) -> np.ndarray:
        """Read all strain snapshots (just one for per-step files).

        Returns shape: (1, n_vertices, 6)
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

    @property
    def green_tile_size_m(self) -> float | None:
        """Spatial tile size in meters. None means use element-count tiling."""
        val = self._file["/simulation"].attrs.get("green_tile_size_m", None)
        return float(val) if val is not None else None


def merge_vertex_records(record_dir: str, n_vertex: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertex-level strain from per-step record files in a directory.

    Discovers all record_{r}_{step}.h5 files in record_dir, groups by step,
    reads per-rank data for each step, and merges by global vertex_id.
    Also supports legacy monolithic record_{r}.h5 files (single-step fallback).

    Args:
        record_dir: Directory containing record_{r}_{step}.h5 files.
        n_vertex: Total number of unique mesh vertices (global).

    Returns:
        (merged_strain, vertex_mask) where
          merged_strain: [n_steps, n_vertex, 6] float32
          vertex_mask:   [n_vertex] bool — True for vertices that were recorded
    """
    import glob
    import re
    import os

    pattern = os.path.join(record_dir, "record_*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No record files found in {record_dir}")

    # Parse filenames and group by step
    # New per-step format: record_{rank}_{step}.h5
    # Legacy monolithic:  record_{rank}.h5  (treated as step 0, single step)
    step_files: dict[int, list[str]] = {}
    has_legacy = False
    for fpath in files:
        basename = os.path.basename(fpath)
        match = re.match(r"record_(\d+)_(\d+)\.h5$", basename)
        if match:
            step = int(match.group(2))
            step_files.setdefault(step, []).append(fpath)
        else:
            match = re.match(r"record_(\d+)\.h5$", basename)
            if match:
                has_legacy = True
                step_files.setdefault(0, []).append(fpath)

    if not step_files:
        raise ValueError(f"No valid record files found in {record_dir}")

    if has_legacy:
        print(
            "[postprocess] Detected legacy monolithic record files (record_{r}.h5).",
            file=sys.stderr,
        )

    sorted_steps = sorted(step_files.keys())

    # Determine dtype from first file
    first_path = step_files[sorted_steps[0]][0]
    with RecordReader(first_path) as r:
        dtype = r.read_strain(0).dtype

    n_steps = len(sorted_steps)
    merged = np.zeros((n_steps, n_vertex, 6), dtype=dtype)
    mask = np.zeros(n_vertex, dtype=bool)

    for snap_idx, step in enumerate(sorted_steps):
        step_merged = np.zeros((n_vertex, 6), dtype=dtype)
        step_mask = np.zeros(n_vertex, dtype=bool)

        for fpath in step_files[step]:
            with RecordReader(fpath) as reader:
                local_ids = reader.vertex_ids
                strain = reader.read_all_strain()  # [1, n_local, 6]
                for local_idx, global_id in enumerate(local_ids):
                    zero_based = int(global_id) - 1
                    if 0 <= zero_based < n_vertex:
                        if step_mask[zero_based]:
                            print(
                                f"[postprocess] WARNING: vertex {global_id} "
                                f"recorded by multiple ranks at step {step} "
                                f"— using last value",
                                file=sys.stderr,
                            )
                        step_merged[zero_based] = strain[0, local_idx]
                        step_mask[zero_based] = True

        merged[snap_idx] = step_merged
        mask |= step_mask

    return merged, mask
