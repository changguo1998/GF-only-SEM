"""GreenFunctionLibrary — top-level entry point for Green's function queries.

Read and index Green's function tiles from HDF5 across one or more SEM
source runs, then route receiver coordinates to the nearest SEM source
via KDTree (reciprocity convention).

Reciprocity convention
----------------------
- ``receiver_xyz`` (the real station) selects **which** Green's function
  run by nearest-neighbour match against SEM source coordinates.
- ``source_xyz`` (the real source) is then looked up among **that** run's
  recorded mesh vertices, interpolated if off-grid.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from greenfun.index_cache import load_or_rebuild_index, SourceIndexEntry
from greenfun.query import GreenQuery
from greenfun.source_run import SourceRun


class GreenFunctionLibrary:
    """Read and index Green's function tiles from HDF5.

    Parameters
    ----------
    root:
        Root directory of the Green's function library, containing
        ``src_XXXX`` subdirectories.
    rebuild_index:
        If True, force a full rescan of all tile files and rebuild the
        index cache, bypassing any existing cache file.
    """

    def __init__(self, root: str | Path, rebuild_index: bool = False) -> None:
        self._root = Path(root).resolve()

        # 1. Load or build library index.
        self._index = load_or_rebuild_index(self._root, rebuild=rebuild_index)

        # 2. Build KDTree over all SEM source coordinates.
        self._sources_by_index: list[SourceIndexEntry] = []
        source_xyz_list: list[npt.NDArray[np.float64]] = []

        for src_entry in sorted(self._index.sources, key=lambda s: s.source_id):
            self._sources_by_index.append(src_entry)
            source_xyz_list.append(src_entry.source_xyz_m)

        if source_xyz_list:
            self._source_xyz_array = np.array(source_xyz_list, dtype=np.float64)
            self._source_kdtree = KDTree(self._source_xyz_array)
        else:
            self._source_xyz_array = np.empty((0, 3), dtype=np.float64)
            self._source_kdtree = None

        # 3. Lazy-loaded SourceRun instances (keyed by source_id).
        self._source_runs: dict[int, SourceRun] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_sources(self) -> int:
        """Number of SEM source runs in the library."""
        return len(self._index.sources)

    @property
    def n_tiles(self) -> int:
        """Total number of tile files across all sources."""
        return len(self._index.tiles)

    def query(
        self,
        source_xyz: npt.ArrayLike,
        receiver_xyz: npt.ArrayLike,
        quantity: str = "strain",
    ) -> GreenQuery:
        """Query Green's function by source and receiver coordinates.

        Parameters
        ----------
        source_xyz:
            Real source coordinate ``[x, y, z]`` in meters.
        receiver_xyz:
            Real receiver (station) coordinate ``[x, y, z]`` in meters.
        quantity:
            One of ``"strain"``, ``"displacement"``, or ``"both"``.

        Returns
        -------
        GreenQuery
            Populated query result.
        """
        receiver = np.asarray(receiver_xyz, dtype=np.float64)
        source = np.asarray(source_xyz, dtype=np.float64)

        # ---- 1. Find nearest SEM source to *receiver* ----
        if self._source_kdtree is None:
            raise ValueError(
                "No SEM sources available. "
                "The library root must contain at least one src_XXXX directory "
                "with tile_*.h5 files."
            )

        distance, idx = self._source_kdtree.query(receiver)
        entry = self._sources_by_index[idx]

        if distance > 1e6:
            warnings.warn(
                f"Large distance ({distance:.1f} m) between receiver "
                f"{receiver} and nearest SEM source at {entry.source_xyz_m}. "
                f"Proceeding anyway.",
                stacklevel=2,
            )

        # ---- 2. Get / lazy-create the matching SourceRun ----
        source_run = self._get_or_create_source_run(entry)

        # ---- 3. Delegate interpolation at *source* ----
        result = source_run.query(source, quantity=quantity)

        # ---- 4. Fix up GreenQuery for reciprocity convention ----
        result.source_xyz = source
        result.receiver_xyz = receiver
        result.sem_source_xyz = entry.source_xyz_m.copy()

        return result

    def query_batch(
        self,
        sources: npt.ArrayLike,
        receivers: npt.ArrayLike,
        quantity: str = "strain",
    ) -> list[GreenQuery]:
        """Batch version of :meth:`query`.

        Parameters
        ----------
        sources:
            Shape ``(n_src, 3)`` — one source per query.
        receivers:
            If shape ``(n_src, 3)``, paired element-wise with *sources*.
            If shape ``(1, 3)`` or ``(3,)``, broadcast to all sources.
        quantity:
            One of ``"strain"``, ``"displacement"``, or ``"both"``.

        Returns
        -------
        list[GreenQuery]
            One result per source.
        """
        src_arr = np.asarray(sources, dtype=np.float64)
        rec_arr = np.asarray(receivers, dtype=np.float64)

        if src_arr.ndim != 2 or src_arr.shape[1] != 3:
            raise ValueError(
                f"sources must have shape (n_src, 3), got {src_arr.shape}"
            )

        n_src = src_arr.shape[0]

        # Reshape receivers for broadcasting.
        if rec_arr.ndim == 1 and rec_arr.shape[0] == 3:
            rec_arr = rec_arr.reshape(1, 3)

        if rec_arr.ndim != 2 or rec_arr.shape[1] != 3:
            raise ValueError(
                f"receivers must have shape (n_rec, 3) or (3,), got {rec_arr.shape}"
            )

        if rec_arr.shape[0] == 1:
            rec_arr = np.broadcast_to(rec_arr, (n_src, 3))

        if rec_arr.shape[0] != n_src:
            raise ValueError(
                f"Number of receivers ({rec_arr.shape[0]}) must match "
                f"number of sources ({n_src}), or be 1 for broadcasting."
            )

        return [
            self.query(src_arr[i], rec_arr[i], quantity=quantity)
            for i in range(n_src)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_source_run(self, entry: SourceIndexEntry) -> SourceRun:
        """Return the cached SourceRun for *entry*, or create it lazily."""
        sid = entry.source_id
        if sid not in self._source_runs:
            dir_path = self._root / entry.dir_path
            self._source_runs[sid] = SourceRun(dir_path, entry.source_xyz_m)
        return self._source_runs[sid]