"""Trilinear interpolation over mesh vertices of a regular Cartesian hex mesh."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree


class TrilinearInterpolator:
    """Trilinear interpolation over a regular Cartesian hex mesh.

    Uses :class:`scipy.spatial.KDTree` for exact-vertex matching and a
    coordinate-axis cell-lookup (via sorted unique axis values) to locate the
    bounding cell of a query point.

    Parameters
    ----------
    vertex_coords:
        Shape ``(n_vertices, 3)``, float64.  Coordinates of every mesh vertex
        of a regular Cartesian (rectilinear) grid.
    """

    def __init__(self, vertex_coords: np.ndarray) -> None:
        coords = np.asarray(vertex_coords, dtype=np.float64)
        self._vertex_coords = coords
        self._tree = KDTree(coords)

        # Sorted unique axis values — enables O(log N) cell lookup.
        x_unique = np.sort(np.unique(coords[:, 0]))
        y_unique = np.sort(np.unique(coords[:, 1]))
        z_unique = np.sort(np.unique(coords[:, 2]))

        self._x_axis = x_unique
        self._y_axis = y_unique
        self._z_axis = z_unique
        self._nx = len(x_unique)
        self._ny = len(y_unique)
        self._nz = len(z_unique)

        # Build mapping (xi, yi, zi) → flat vertex index.
        cell_map = np.full((self._nx, self._ny, self._nz), -1, dtype=np.intp)
        for vtx_idx in range(coords.shape[0]):
            x, y, z = coords[vtx_idx]
            xi = int(np.searchsorted(x_unique, x))
            yi = int(np.searchsorted(y_unique, y))
            zi = int(np.searchsorted(z_unique, z))
            cell_map[xi, yi, zi] = vtx_idx
        self._cell_map = cell_map

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate(
        self,
        point_xyz_m: npt.ArrayLike,
        values: np.ndarray,
    ) -> np.ndarray:
        """Interpolate *values* at the query point.

        Parameters
        ----------
        point_xyz_m:
            Query coordinate, shape ``(3,)``.
        values:
            Vertex-valued data, shape ``(n_vertices,)`` or
            ``(n_vertices, ...)``.  The first dimension must equal the number
            of vertices passed at construction time.

        Returns
        -------
        Interpolated value(s) with the trailing dimensions of *values*
        preserved unchanged.

        Raises
        ------
        ValueError
            If the point lies outside the mesh bounds, or if shape mismatches
            are detected.
        """
        point = np.asarray(point_xyz_m, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(
                f"point_xyz_m must have shape (3,), got {point.shape}"
            )

        values = np.asarray(values)
        n_vertices = self._vertex_coords.shape[0]
        if values.shape[0] != n_vertices:
            raise ValueError(
                f"values first dimension ({values.shape[0]}) does not match "
                f"n_vertices ({n_vertices})"
            )

        # ------------------------------------------------------------------
        # 1.  Quick exact-vertex check via KDTree (handles boundary vertices
        #     that the cell-lookup below would reject as out-of-bounds).
        # ------------------------------------------------------------------
        nn_distance, nn_index = self._tree.query(point, k=1)
        if nn_distance < 1e-15:
            return values[int(nn_index)]

        # ------------------------------------------------------------------
        # 2.  Bounds check & cell location via axis search
        # ------------------------------------------------------------------
        x_axis, y_axis, z_axis = self._x_axis, self._y_axis, self._z_axis
        px, py, pz = float(point[0]), float(point[1]), float(point[2])

        ix = int(np.searchsorted(x_axis, px, side="right")) - 1
        iy = int(np.searchsorted(y_axis, py, side="right")) - 1
        iz = int(np.searchsorted(z_axis, pz, side="right")) - 1

        if ix < 0 or iy < 0 or iz < 0:
            raise ValueError(
                f"Query point {point} is outside mesh bounds (below minimum)."
            )
        if ix >= self._nx - 1 or iy >= self._ny - 1 or iz >= self._nz - 1:
            raise ValueError(
                f"Query point {point} is outside mesh bounds (above maximum)."
            )

        # 8 corners of the bounding cell
        corner_indices = [
            self._cell_map[ix + i, iy + j, iz + k]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]

        # ------------------------------------------------------------------
        # 3.  Cell bounds
        # ------------------------------------------------------------------
        x0, x1 = x_axis[ix], x_axis[ix + 1]
        y0, y1 = y_axis[iy], y_axis[iy + 1]
        z0, z1 = z_axis[iz], z_axis[iz + 1]

        cell_size = np.array(
            [x1 - x0, y1 - y0, z1 - z0], dtype=np.float64
        )

        # Degenerate cell — use inverse-distance weighting.
        if np.any(cell_size <= 1e-15):
            distances, _ = self._tree.query(point, k=8)
            weights = 1.0 / (distances + 1e-300)
            weights /= weights.sum()
            corner_values = values[corner_indices]
            return np.tensordot(weights, corner_values, axes=(0, 0))

        # ------------------------------------------------------------------
        # 4.  Local coordinates (alpha, beta, gamma) ∈ [0, 1]
        # ------------------------------------------------------------------
        alpha = np.clip((px - x0) / cell_size[0], 0.0, 1.0)
        beta = np.clip((py - y0) / cell_size[1], 0.0, 1.0)
        gamma = np.clip((pz - z0) / cell_size[2], 0.0, 1.0)

        # ------------------------------------------------------------------
        # 5.  Trilinear weights
        #     w(i,j,k) = (1-α)^{1-i}·α^i · (1-β)^{1-j}·β^j · (1-γ)^{1-k}·γ^k
        # ------------------------------------------------------------------
        weights = np.empty(8, dtype=np.float64)
        for c in range(8):
            i, j, k = c // 4, (c // 2) % 2, c % 2
            weights[c] = (
                (1.0 - alpha) ** (1 - i) * alpha ** i
                * (1.0 - beta) ** (1 - j) * beta ** j
                * (1.0 - gamma) ** (1 - k) * gamma ** k
            )

        weights /= weights.sum()

        # ------------------------------------------------------------------
        # 6.  Weighted sum
        # ------------------------------------------------------------------
        corner_values = values[corner_indices]  # (8, ...)
        return np.tensordot(weights, corner_values, axes=(0, 0))