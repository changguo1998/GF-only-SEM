"""SourceRun — one SEM source run with lazy tile loading and interpolation.

A SourceRun encapsulates all tiles for a single SEM source location. It
discovers ``tile_*.h5`` files, reads vertex coordinates, deduplicates
boundary vertices by ID, and provides interpolation via
:class:`TrilinearInterpolator`.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from greenfun.gll_interpolator import EXACT_GLL_NODE_TOLERANCE_M, GLLInterpolator
from greenfun.interpolator import EXACT_VERTEX_TOLERANCE_M, TrilinearInterpolator
from greenfun.query import GreenQuery


class SourceRun:
    """Single greenfun run (one SEM source). Lazy tile loading.

    Parameters
    ----------
    dir_path:
        Path to a ``src_XXXX`` directory containing ``tile_*.h5`` files.
    source_xyz:
        SEM source location ``[x, y, z]`` in meters.

    Attributes
    ----------
    time:
        Time axis ``[nt]``, populated after :meth:`load`.
    vertex_coords:
        Deduplicated vertex coordinates ``[n_vertices, 3]``.
    greens_tensor:
        Strain Green tensor ``[nt, n_vertices, 6, 3]``.
    displacement_tensor:
        Displacement tensor ``[nt, n_vertices, 3, 3]`` or ``None``.
    n_tiles:
        Number of tile files loaded.
    """

    def __init__(self, dir_path: Path, source_xyz: np.ndarray) -> None:
        self._dir_path = Path(dir_path)
        self._source_xyz = np.asarray(source_xyz, dtype=np.float64)

        # Lazy-loaded state
        self._loaded = False
        self.time: npt.NDArray[np.float64] | None = None
        self.vertex_coords: npt.NDArray[np.float64] | None = None
        self.greens_tensor: npt.NDArray[np.float32] | None = None
        self.displacement_tensor: npt.NDArray[np.float32] | None = None
        self.velocity_tensor: npt.NDArray[np.float32] | None = None
        self.acceleration_tensor: npt.NDArray[np.float32] | None = None
        self.n_tiles: int = 0
        self._gll_mode: bool = False
        self._interpolator: TrilinearInterpolator | GLLInterpolator | None = None
        self._kdtree: KDTree | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all tiles, detecting basis and reading accordingly.

        Raises
        ------
        FileNotFoundError
            If no ``tile_*.h5`` files are found in the directory.
        """
        tile_paths = sorted(self._dir_path.glob("tile_*.h5"))
        if not tile_paths:
            raise FileNotFoundError(f"No tile_*.h5 files found in {self._dir_path}")

        # Read time axis + basis from the first tile.
        with h5py.File(tile_paths[0], "r") as h5:
            self.time = h5["/time/t"][:].astype(np.float64)
            basis = h5.attrs.get("basis", "mesh_vertices")

        nt = len(self.time)
        self._gll_mode = basis == "gll"

        if self._gll_mode:
            self._load_gll_tiles(tile_paths, nt)
        else:
            self._load_vertex_tiles(tile_paths, nt)

        self.n_tiles = len(tile_paths)
        self._kdtree = KDTree(self.vertex_coords)
        self._loaded = True

    # ------------------------------------------------------------------
    # Internal tile loaders
    # ------------------------------------------------------------------

    def _load_vertex_tiles(self, tile_paths: list[Path], nt: int) -> None:
        """Load vertex-based (corner) tiles with cross-tile deduplication."""
        all_vertex_ids: list[np.ndarray] = []
        all_vertex_coords: list[np.ndarray] = []
        all_greens: list[np.ndarray] = []
        all_displacements: list[np.ndarray] = []
        seen_ids: set[int] = set()
        all_velocities: list[np.ndarray] = []
        all_accelerations: list[np.ndarray] = []
        has_velocity = False
        has_acceleration = False
        has_displacement = False

        for tile_path in tile_paths:
            with h5py.File(tile_path, "r") as h5:
                vertex_ids = h5["/mesh/vertex_ids"][:]
                vertex_coords = h5["/mesh/vertex_coords"][:]
                greens = h5["/field/greens_tensor"][:]

                has_vel = "/field/velocity_tensor" in h5
                if has_vel:
                    velocity = h5["/field/velocity_tensor"][:]
                    has_velocity = True
                else:
                    velocity = None

                has_acc = "/field/acceleration_tensor" in h5
                if has_acc:
                    acceleration = h5["/field/acceleration_tensor"][:]
                    has_acceleration = True
                else:
                    acceleration = None

                has_disp = "/field/displacement_tensor" in h5
                if has_disp:
                    displacement = h5["/field/displacement_tensor"][:]
                    has_displacement = True
                else:
                    displacement = None

            # Deduplicate: keep vertices whose ID has not been seen before.
            keep_mask = np.array([int(vid) not in seen_ids for vid in vertex_ids], dtype=bool)
            kept_ids = vertex_ids[keep_mask]
            for vid in kept_ids:
                seen_ids.add(int(vid))

            if not np.any(keep_mask):
                continue

            all_vertex_ids.append(kept_ids)
            all_vertex_coords.append(vertex_coords[keep_mask])
            all_greens.append(greens[:, keep_mask, :, :])
            if displacement is not None:
                all_displacements.append(displacement[:, keep_mask, :, :])
            if velocity is not None:
                all_velocities.append(velocity[:, keep_mask, :, :])
            if acceleration is not None:
                all_accelerations.append(acceleration[:, keep_mask, :, :])

        self.vertex_coords = np.concatenate(all_vertex_coords, axis=0).astype(np.float64)
        self.greens_tensor = np.concatenate(all_greens, axis=1).astype(np.float32)

        if has_displacement and all_displacements:
            self.displacement_tensor = np.concatenate(all_displacements, axis=1).astype(np.float32)
        else:
            self.displacement_tensor = None

        if has_velocity and all_velocities:
            self.velocity_tensor = np.concatenate(all_velocities, axis=1).astype(np.float32)
        else:
            self.velocity_tensor = None

        if has_acceleration and all_accelerations:
            self.acceleration_tensor = np.concatenate(all_accelerations, axis=1).astype(np.float32)
        else:
            self.acceleration_tensor = None

        self._interpolator = TrilinearInterpolator(self.vertex_coords)

    def _load_gll_tiles(self, tile_paths: list[Path], nt: int) -> None:
        """Load GLL-based tiles — no dedup (postprocess already does it).

        GLL tiles already store unique nodes per tile and tiles are disjoint.
        Simply concatenate across tiles.
        """
        all_node_ids: list[np.ndarray] = []
        all_node_coords: list[np.ndarray] = []
        all_cell_gll_index: list[np.ndarray] = []
        all_greens: list[np.ndarray] = []
        all_displacements: list[np.ndarray] = []
        all_velocities: list[np.ndarray] = []
        all_accelerations: list[np.ndarray] = []
        has_displacement = False
        has_velocity = False
        has_acceleration = False

        for tile_path in tile_paths:
            with h5py.File(tile_path, "r") as h5:
                node_ids = h5["/mesh/gll_node_ids"][:]
                node_coords = h5["/mesh/gll_node_coords"][:]
                cell_index = h5["/mesh/cell_gll_node_index"][:]
                greens = h5["/field/greens_tensor"][:]

                has_disp = "/field/displacement_tensor" in h5
                if has_disp:
                    displacement = h5["/field/displacement_tensor"][:]
                    has_displacement = True
                else:
                    displacement = None

                has_vel = "/field/velocity_tensor" in h5
                if has_vel:
                    velocity = h5["/field/velocity_tensor"][:]
                    has_velocity = True
                else:
                    velocity = None

                has_acc = "/field/acceleration_tensor" in h5
                if has_acc:
                    acceleration = h5["/field/acceleration_tensor"][:]
                    has_acceleration = True
                else:
                    acceleration = None

            all_node_ids.append(node_ids)
            all_node_coords.append(node_coords)
            all_cell_gll_index.append(cell_index)
            all_greens.append(greens[:, :, :, :])
            if displacement is not None:
                all_displacements.append(displacement[:, :, :, :])
            if velocity is not None:
                all_velocities.append(velocity[:, :, :, :])
            if acceleration is not None:
                all_accelerations.append(acceleration[:, :, :, :])

        self.vertex_coords = np.concatenate(all_node_coords, axis=0).astype(np.float64)
        self.greens_tensor = np.concatenate(all_greens, axis=1).astype(np.float32)

        if has_displacement and all_displacements:
            self.displacement_tensor = np.concatenate(all_displacements, axis=1).astype(np.float32)
        else:
            self.displacement_tensor = None

        if has_velocity and all_velocities:
            self.velocity_tensor = np.concatenate(all_velocities, axis=1).astype(np.float32)
        else:
            self.velocity_tensor = None

        if has_acceleration and all_accelerations:
            self.acceleration_tensor = np.concatenate(all_accelerations, axis=1).astype(np.float32)
        else:
            self.acceleration_tensor = None

        cell_gll_node_index = np.concatenate(all_cell_gll_index, axis=0)
        self._interpolator = GLLInterpolator(
            gll_node_coords=self.vertex_coords,
            cell_gll_node_index=cell_gll_node_index,
        )

    def query(self, source_xyz_m: npt.ArrayLike, quantity: str = "strain") -> GreenQuery:
        """Return interpolated Green's function at *source_xyz_m*.

        Parameters
        ----------
        source_xyz_m:
            Real source coordinate ``[x, y, z]`` in meters.
        quantity:
            One of ``"strain"``, ``"displacement"``, ``"velocity"``, ``"acceleration"``, or ``"both"``.

        Returns
        -------
        GreenQuery
            Fully populated query result.
        """
        if quantity not in ("strain", "displacement", "velocity", "acceleration", "both"):
            raise ValueError(
                f"quantity must be one of 'strain', 'displacement', 'velocity', "
                f"'acceleration', or 'both', got {quantity!r}"
            )
        if not self._loaded:
            self.load()

        point = np.asarray(source_xyz_m, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"source_xyz_m must have shape (3,), got {point.shape}")

        # Check for exact vertex match (tolerance absorbs float64 rounding at large coords).
        nn_distance, _ = self._kdtree.query(point, k=1)  # type: ignore[union-attr]
        interpolation_used = bool(nn_distance >= EXACT_VERTEX_TOLERANCE_M)

        # Interpolate requested quantities.
        # Tensors are stored as [nt, n_vertices, ...] but TrilinearInterpolator
        # expects [n_vertices, ...], so we move the vertex dimension first.
        strain: npt.NDArray[np.float32] | None = None
        displacement: npt.NDArray[np.float32] | None = None
        velocity: npt.NDArray[np.float32] | None = None
        acceleration: npt.NDArray[np.float32] | None = None

        if quantity in ("strain", "both"):
            values_vtx_first = np.moveaxis(self.greens_tensor, 0, 1)  # -> [n_vertices, nt, 6, 3]
            result = self._interpolator.interpolate(  # type: ignore[union-attr]
                point, values_vtx_first
            )
            strain = np.asarray(result, dtype=np.float32)  # [nt, 6, 3]

        if quantity in ("displacement", "both"):
            if self.displacement_tensor is not None:
                values_vtx_first = np.moveaxis(
                    self.displacement_tensor, 0, 1
                )  # -> [n_vertices, nt, 3, 3]
                result = self._interpolator.interpolate(  # type: ignore[union-attr]
                    point, values_vtx_first
                )
                displacement = np.asarray(result, dtype=np.float32)  # [nt, 3, 3]

        if quantity in ("velocity", "both"):
            if self.velocity_tensor is not None:
                values_vtx_first = np.moveaxis(self.velocity_tensor, 0, 1)
                result = self._interpolator.interpolate(point, values_vtx_first)
                velocity = np.asarray(result, dtype=np.float32)

        if quantity in ("acceleration", "both"):
            if self.acceleration_tensor is not None:
                values_vtx_first = np.moveaxis(self.acceleration_tensor, 0, 1)
                result = self._interpolator.interpolate(point, values_vtx_first)
                acceleration = np.asarray(result, dtype=np.float32)

        return GreenQuery(
            source_xyz=point,
            receiver_xyz=point,
            sem_source_xyz=self._source_xyz,
            time=self.time,  # type: ignore[arg-type]
            strain=strain,
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
            n_tiles_used=self.n_tiles,
            interpolation_used=interpolation_used,
        )
