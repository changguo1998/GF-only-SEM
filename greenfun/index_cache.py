"""Index cache for greenfun library — rebuildable HDF5 cache of tile metadata.

Provides:
- ``compute_library_hash`` — blake2b hash of tile files (stat only, no HDF5 I/O).
- ``scan_tiles`` — open every tile, read attrs, build a fresh ``LibraryIndex``.
- ``load_or_rebuild_index`` — load cached index if hash matches, otherwise rebuild.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import re
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt

CACHE_FILENAME = "_greenfun_index.h5"
"""Name of the cache file stored at the library root."""

_CACHE_VERSION = "1.0"
"""Cache schema version."""

_SRC_DIR_RE = re.compile(r"src_(\d+)$")
"""Match ``src_NNNN`` directory names and extract the integer suffix."""


# ── data classes ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class SourceIndexEntry:
    """Metadata for one SEM-source directory in the library."""

    source_id: int
    dir_path: str
    source_xyz_m: npt.NDArray[np.float64]
    n_tiles: int


@dataclasses.dataclass
class TileIndexEntry:
    """Metadata for one tile HDF5 file in the library."""

    source_id: int
    rel_path: str
    tile_ij: tuple[int, int]
    bounds_m: npt.NDArray[np.float64]


@dataclasses.dataclass
class LibraryIndex:
    """Full in-memory index of a greenfun library."""

    sources: list[SourceIndexEntry]
    tiles: list[TileIndexEntry]
    library_hash: str
    build_time: str


# ── hash computation ─────────────────────────────────────────────────────


def compute_library_hash(root_path: Path) -> str:
    """Compute a blake2b hash of the current tile set.

    The hash covers ``(rel_path, size_bytes, mtime_ns)`` for every
    ``**/tile_*.h5`` under *root_path*, sorted lexicographically by
    *rel_path*.  Only ``os.stat`` is performed — no HDF5 I/O.

    Returns
    -------
    str
        Hex-encoded blake2b digest.
    """
    root_path = Path(root_path).resolve()
    entries: list[tuple[str, int, int]] = []

    for tile_path in sorted(root_path.rglob("tile_*.h5")):
        rel = str(tile_path.relative_to(root_path))
        stat = tile_path.stat()
        entries.append((rel, stat.st_size, stat.st_mtime_ns))

    hasher = hashlib.blake2b()
    for rel, size, mtime_ns in entries:
        hasher.update(rel.encode("utf-8"))
        hasher.update(size.to_bytes(8, "little"))
        hasher.update(mtime_ns.to_bytes(8, "little"))

    return hasher.hexdigest()


# ── tile scanning (full rebuild) ────────────────────────────────────────


def _extract_source_id(source_dir: Path, root_path: Path) -> int:
    """Extract the integer source ID from a source directory name."""
    match = _SRC_DIR_RE.search(source_dir.name)
    if match is None:
        raise ValueError(
            f"Source directory {source_dir.name!r} does not match pattern 'src_NNNN'"
        )
    return int(match.group(1))


def _read_tile_attrs(tile_path: Path) -> TileIndexEntry:
    """Open a single tile HDF5 and extract its index metadata."""
    with h5py.File(tile_path, "r") as h5:
        source_xyz = h5.attrs["source_xyz_m"]
        if isinstance(source_xyz, np.ndarray):
            source_xyz = source_xyz.astype(np.float64)
        else:
            source_xyz = np.array(source_xyz, dtype=np.float64)

        min_max = {
            k: float(h5.attrs[k])
            for k in (
                "x_min_m",
                "x_max_m",
                "y_min_m",
                "y_max_m",
                "z_min_m",
                "z_max_m",
            )
        }
        bounds = np.array(
            [
                min_max["x_min_m"],
                min_max["x_max_m"],
                min_max["y_min_m"],
                min_max["y_max_m"],
                min_max["z_min_m"],
                min_max["z_max_m"],
            ],
            dtype=np.float64,
        )
        tile_x = int(h5.attrs["tile_x_index"])
        tile_y = int(h5.attrs["tile_y_index"])

    # Derive source identity from the tile's parent directory.
    source_dir = tile_path.parent
    source_id = _extract_source_id(source_dir, tile_path.parent)

    # rel_path is relative to the library root (grandparent of tile).
    root_path = tile_path.parent.parent
    rel_path = str(tile_path.relative_to(root_path))

    return TileIndexEntry(
        source_id=source_id,
        rel_path=rel_path,
        tile_ij=(tile_x, tile_y),
        bounds_m=bounds,
    )


def scan_tiles(root_path: Path) -> LibraryIndex:
    """Scan every tile file in the library and build a fresh index.

    Each tile HDF5 is opened to read its ``source_xyz_m`` and bounds
    attributes.  This is the expensive path (all tiles must be opened).

    Returns
    -------
    LibraryIndex
        Freshly-scanned index with an empty ``library_hash`` (caller should
        compute and set it via :func:`compute_library_hash`).
    """
    root_path = Path(root_path).resolve()
    tile_paths = sorted(root_path.rglob("tile_*.h5"))

    # Parse every tile.
    tile_entries: list[TileIndexEntry] = []
    seen_sources: dict[int, npt.NDArray[np.float64]] = {}

    for tile_path in tile_paths:
        entry = _read_tile_attrs(tile_path)
        tile_entries.append(entry)

        # Stash the source xyz on first encounter of each source.
        if entry.source_id not in seen_sources:
            # Re-read the tile to grab source_xyz_m.
            with h5py.File(tile_path, "r") as h5:
                seen_sources[entry.source_id] = np.asarray(
                    h5.attrs["source_xyz_m"], dtype=np.float64
                )

    # Build source entries (one per unique source directory).
    # Source directories are discovered from tile parent dirs.
    source_dirs: dict[int, Path] = {}
    for tile_path in tile_paths:
        sid = _extract_source_id(tile_path.parent, root_path)
        if sid not in source_dirs:
            source_dirs[sid] = tile_path.parent

    source_entries: list[SourceIndexEntry] = []
    for sid in sorted(source_dirs):
        sdir = source_dirs[sid]
        dir_rel = str(sdir.relative_to(root_path))
        n_t = sum(1 for t in tile_entries if t.source_id == sid)
        xyz = seen_sources.get(
            sid, np.array([0.0, 0.0, 0.0], dtype=np.float64)
        )
        source_entries.append(
            SourceIndexEntry(
                source_id=sid,
                dir_path=dir_rel,
                source_xyz_m=xyz,
                n_tiles=n_t,
            )
        )

    build_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return LibraryIndex(
        sources=source_entries,
        tiles=tile_entries,
        library_hash="",  # caller fills this in
        build_time=build_time,
    )


# ── cache I/O ────────────────────────────────────────────────────────────


def _write_cache(cache_path: Path, index: LibraryIndex) -> None:
    """Write a ``LibraryIndex`` to the HDF5 cache file."""
    with h5py.File(cache_path, "w") as h5:
        # Root attrs.
        h5.attrs["version"] = _CACHE_VERSION
        h5.attrs["library_hash"] = index.library_hash
        h5.attrs["build_time"] = index.build_time
        h5.attrs["n_sources"] = len(index.sources)
        h5.attrs["n_tiles"] = len(index.tiles)

        # Sources group.
        src_dtype = np.dtype(
            [
                ("source_id", np.int32),
                ("source_xyz_m", np.float64, (3,)),
                ("n_tiles", np.int32),
            ]
        )
        src_data = np.empty(len(index.sources), dtype=src_dtype)
        src_dir_paths: list[str] = []
        for i, s in enumerate(index.sources):
            src_data["source_id"][i] = s.source_id
            src_data["source_xyz_m"][i] = s.source_xyz_m
            src_data["n_tiles"][i] = s.n_tiles
            src_dir_paths.append(s.dir_path)

        grp_src = h5.create_group("sources")
        grp_src.create_dataset("source_id", data=src_data["source_id"])
        grp_src.create_dataset(
            "source_xyz_m", data=src_data["source_xyz_m"]
        )
        grp_src.create_dataset("n_tiles", data=src_data["n_tiles"])

        # Variable-length string dataset for dir_path.
        dt_str = h5py.string_dtype()
        grp_src.create_dataset(
            "dir_path",
            data=np.array(src_dir_paths, dtype=object),
            dtype=dt_str,
        )

        # Tiles group.
        n_tiles = len(index.tiles)
        tile_source_ids = np.empty(n_tiles, dtype=np.int32)
        tile_rel_paths: list[str] = []
        tile_ij = np.empty((n_tiles, 2), dtype=np.int32)
        tile_bounds = np.empty((n_tiles, 6), dtype=np.float64)

        for i, t in enumerate(index.tiles):
            tile_source_ids[i] = t.source_id
            tile_rel_paths.append(t.rel_path)
            tile_ij[i, 0] = t.tile_ij[0]
            tile_ij[i, 1] = t.tile_ij[1]
            tile_bounds[i, :] = t.bounds_m

        grp_tile = h5.create_group("tiles")
        grp_tile.create_dataset("source_id", data=tile_source_ids)
        grp_tile.create_dataset(
            "rel_path",
            data=np.array(tile_rel_paths, dtype=object),
            dtype=dt_str,
        )
        grp_tile.create_dataset("tile_ij", data=tile_ij)
        grp_tile.create_dataset("bounds_m", data=tile_bounds)


def _load_cache(
    cache_path: Path, library_hash: str
) -> LibraryIndex | None:
    """Load the cached index if the hash matches.

    Returns ``None`` when the cache is missing, has an incompatible version,
    or the stored hash does not match *library_hash*.
    """
    if not cache_path.is_file():
        return None

    with h5py.File(cache_path, "r") as h5:
        # Version check.
        stored_version = str(h5.attrs.get("version", ""))
        if stored_version != _CACHE_VERSION:
            return None

        stored_hash = str(h5.attrs.get("library_hash", ""))
        if stored_hash != library_hash:
            return None

        build_time = str(h5.attrs.get("build_time", ""))

        # Read sources.
        grp_src = h5["sources"]
        src_ids = grp_src["source_id"][:]
        src_xyz = grp_src["source_xyz_m"][:]
        src_n_tiles = grp_src["n_tiles"][:]
        src_dir_paths = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in grp_src["dir_path"][:]
        ]

        sources: list[SourceIndexEntry] = []
        for i in range(len(src_ids)):
            sources.append(
                SourceIndexEntry(
                    source_id=int(src_ids[i]),
                    dir_path=src_dir_paths[i],
                    source_xyz_m=np.asarray(src_xyz[i], dtype=np.float64),
                    n_tiles=int(src_n_tiles[i]),
                )
            )

        # Read tiles.
        grp_tile = h5["tiles"]
        tile_src_ids = grp_tile["source_id"][:]
        tile_rel_paths = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in grp_tile["rel_path"][:]
        ]
        tile_ij = grp_tile["tile_ij"][:]
        tile_bounds = grp_tile["bounds_m"][:]

        tiles: list[TileIndexEntry] = []
        for i in range(len(tile_src_ids)):
            tiles.append(
                TileIndexEntry(
                    source_id=int(tile_src_ids[i]),
                    rel_path=tile_rel_paths[i],
                    tile_ij=(int(tile_ij[i, 0]), int(tile_ij[i, 1])),
                    bounds_m=np.asarray(tile_bounds[i], dtype=np.float64),
                )
            )

    return LibraryIndex(
        sources=sources,
        tiles=tiles,
        library_hash=library_hash,
        build_time=build_time,
    )


# ── public entry point ───────────────────────────────────────────────────


def load_or_rebuild_index(
    root_path: Path, rebuild: bool = False
) -> LibraryIndex:
    """Load the cached index or rebuild it from scratch.

    Parameters
    ----------
    root_path : Path
        Root directory of the greenfun library.
    rebuild : bool
        If ``True``, skip the cache and force a full rescan.

    Returns
    -------
    LibraryIndex
        Index with ``library_hash`` and ``build_time`` populated.

    Notes
    -----
    The startup flow is:

    1. Glob ``root/**/tile_*.h5``, stat each, compute ``current_hash``.
    2. Try to read ``_greenfun_index.h5``.
    3. If hash matches → load and return cached index.
    4. Otherwise (mismatch / missing / rebuild=True) → scan all tiles,
       write new cache, return.
    """
    root_path = Path(root_path).resolve()
    cache_path = root_path / CACHE_FILENAME

    # Step 1: compute hash of current tile set.
    current_hash = compute_library_hash(root_path)

    # Step 2 & 3: try cache hit.
    if not rebuild:
        cached = _load_cache(cache_path, current_hash)
        if cached is not None:
            return cached

    # Step 4: rebuild.
    index = scan_tiles(root_path)
    index.library_hash = current_hash
    _write_cache(cache_path, index)
    return index