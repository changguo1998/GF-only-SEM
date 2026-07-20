"""GLL Lagrange interpolation for Green's function tiles with basis='gll'.

For GLL tiles, each tile stores unique GLL nodes (L2-projected in postprocess)
plus ``cell_gll_node_index`` mapping each recording cell to its 125 GLL nodes.
This interpolator locates the cell containing a query point, maps to reference
coordinates (xi, eta, zeta) in [-1, 1]^3, and evaluates the 3D tensor-product
GLL Lagrange basis for spectral-accuracy interpolation.

Exact GLL-node matches are returned directly (zero interpolation error) via
KDTree lookup.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

# Tolerance for treating a query point as an exact GLL-node match (1 micron).
EXACT_GLL_NODE_TOLERANCE_M = 1e-6

# 1D GLL points for N=4 (5 nodes) on the reference interval [-1, 1].
# Roots of (1 - xi^2) * P'_N(xi) = 0 where P_N is the Legendre polynomial.
_NGLL = 5
_GLL_XI_1D = np.array([-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0], dtype=np.float64)

# Indices of the 8 corner nodes within a cell's 125-node flat array
# (layout: i*NGLL*NGLL + j*NGLL + k, x-major, y-middle, z-minor).
_CORNER_LOCAL_INDICES = np.array(
    [
        0 * _NGLL * _NGLL + 0 * _NGLL + 0,  # (0,0,0) = index   0
        0 * _NGLL * _NGLL + 0 * _NGLL + 4,  # (0,0,4) = index   4
        0 * _NGLL * _NGLL + 4 * _NGLL + 0,  # (0,4,0) = index  20
        0 * _NGLL * _NGLL + 4 * _NGLL + 4,  # (0,4,4) = index  24
        4 * _NGLL * _NGLL + 0 * _NGLL + 0,  # (4,0,0) = index 100
        4 * _NGLL * _NGLL + 0 * _NGLL + 4,  # (4,0,4) = index 104
        4 * _NGLL * _NGLL + 4 * _NGLL + 0,  # (4,4,0) = index 120
        4 * _NGLL * _NGLL + 4 * _NGLL + 4,  # (4,4,4) = index 124
    ],
    dtype=np.intp,
)


class GLLInterpolator:
    """Spectral GLL Lagrange interpolation over recording cells.

    Parameters
    ----------
    gll_node_coords:
        Shape ``(n_unique_gll, 3)``, float64.  Coordinates of each unique GLL
        node in the tile (deduplicated by postprocess L2 projection).
    cell_gll_node_index:
        Shape ``(n_recording_cell, 125)``, int32/int64.  For each recording
        cell, the indices (into *gll_node_coords*) of its 125 GLL nodes,
        in standard 3D tensor-product order (x-major, y-middle, z-minor).
    ngll:
        Number of GLL nodes per dimension (default 5 for N=4 polynomial).
    """

    def __init__(
        self, gll_node_coords: np.ndarray, cell_gll_node_index: np.ndarray, ngll: int = 5
    ) -> None:
        if ngll != 5:
            raise NotImplementedError("Only ngll=5 (N=4) is currently supported")

        self._ngll = ngll
        self._gll_node_coords = np.asarray(gll_node_coords, dtype=np.float64)
        self._cell_gll_node_index = np.asarray(cell_gll_node_index, dtype=np.int64)

        n_unique = self._gll_node_coords.shape[0]
        n_cell = self._cell_gll_node_index.shape[0]

        if self._cell_gll_node_index.shape[1] != ngll**3:
            raise ValueError(
                f"cell_gll_node_index must have shape (n_cell, {ngll**3}), "
                f"got {self._cell_gll_node_index.shape}"
            )

        # KDTree for exact-node-match queries.
        self._node_tree = KDTree(self._gll_node_coords)

        # Precompute cell centers + bounding boxes for cell lookup.
        self._cell_centers = np.empty((n_cell, 3), dtype=np.float64)
        self._cell_x_min = np.empty(n_cell, dtype=np.float64)
        self._cell_x_max = np.empty(n_cell, dtype=np.float64)
        self._cell_y_min = np.empty(n_cell, dtype=np.float64)
        self._cell_y_max = np.empty(n_cell, dtype=np.float64)
        self._cell_z_min = np.empty(n_cell, dtype=np.float64)
        self._cell_z_max = np.empty(n_cell, dtype=np.float64)

        for c in range(n_cell):
            node_idx = self._cell_gll_node_index[c, _CORNER_LOCAL_INDICES]
            corners = self._gll_node_coords[node_idx]
            xc = corners[:, 0]
            yc = corners[:, 1]
            zc = corners[:, 2]
            self._cell_x_min[c] = xc.min()
            self._cell_x_max[c] = xc.max()
            self._cell_y_min[c] = yc.min()
            self._cell_y_max[c] = yc.max()
            self._cell_z_min[c] = zc.min()
            self._cell_z_max[c] = zc.max()
            self._cell_centers[c, 0] = 0.5 * (self._cell_x_min[c] + self._cell_x_max[c])
            self._cell_centers[c, 1] = 0.5 * (self._cell_y_min[c] + self._cell_y_max[c])
            self._cell_centers[c, 2] = 0.5 * (self._cell_z_min[c] + self._cell_z_max[c])

        self._cell_tree = KDTree(self._cell_centers)

        # Precompute GLL Lagrange basis for each 1D reference point *at each
        # GLL node* — used to verify that interpolation reproduces node values.
        # Not needed for interpolation itself (which evaluates at arbitrary xi).

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def interpolate(self, point_xyz_m: npt.ArrayLike, values: np.ndarray) -> np.ndarray:
        """Interpolate *values* at the query point via GLL Lagrange basis.

        Parameters
        ----------
        point_xyz_m:
            Query coordinate, shape ``(3,)``.
        values:
            Node-valued data, shape ``(n_unique_gll,)`` or
            ``(n_unique_gll, ...)``.  First dimension must equal the number
            of unique GLL nodes.

        Returns
        -------
        Interpolated value(s) with trailing dimensions preserved.

        Raises
        ------
        ValueError
            If the point lies outside all recording cells, or if shape
            mismatches are detected.
        """
        point = np.asarray(point_xyz_m, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"point_xyz_m must have shape (3,), got {point.shape}")

        values = np.asarray(values)
        n_unique = self._gll_node_coords.shape[0]
        if values.shape[0] != n_unique:
            raise ValueError(
                f"values first dimension ({values.shape[0]}) does not match "
                f"n_unique_gll ({n_unique})"
            )

        # ------------------------------------------------------------------
        # 1. Exact GLL-node match via KDTree.
        # ------------------------------------------------------------------
        nn_distance, nn_index = self._node_tree.query(point, k=1)
        if nn_distance < EXACT_GLL_NODE_TOLERANCE_M:
            return np.asarray(values[int(nn_index)])

        # ------------------------------------------------------------------
        # 2. Cell lookup via cell-center KDTree.
        # ------------------------------------------------------------------
        px, py, pz = float(point[0]), float(point[1]), float(point[2])

        # Find candidate cells (check nearest K in case of boundary).
        _, candidate_indices = self._cell_tree.query(point, k=min(8, len(self._cell_centers)))

        # Ensure scalar index handling for k=1.
        if isinstance(candidate_indices, np.integer):
            candidate_indices = np.array([candidate_indices])

        cell_idx = -1
        for ci in candidate_indices:
            ci = int(ci)
            if (
                self._cell_x_min[ci] - EXACT_GLL_NODE_TOLERANCE_M
                <= px
                <= self._cell_x_max[ci] + EXACT_GLL_NODE_TOLERANCE_M
                and self._cell_y_min[ci] - EXACT_GLL_NODE_TOLERANCE_M
                <= py
                <= self._cell_y_max[ci] + EXACT_GLL_NODE_TOLERANCE_M
                and self._cell_z_min[ci] - EXACT_GLL_NODE_TOLERANCE_M
                <= pz
                <= self._cell_z_max[ci] + EXACT_GLL_NODE_TOLERANCE_M
            ):
                cell_idx = ci
                break

        if cell_idx < 0:
            raise ValueError(
                f"Query point {point} is outside all recording cells. "
                f"Cell bounds: x=[{self._cell_x_min.min():.1f}, {self._cell_x_max.max():.1f}], "
                f"y=[{self._cell_y_min.min():.1f}, {self._cell_y_max.max():.1f}], "
                f"z=[{self._cell_z_min.min():.1f}, {self._cell_z_max.max():.1f}]"
            )

        # ------------------------------------------------------------------
        # 3. Reference coordinate mapping (physical → [-1, 1]^3).
        # ------------------------------------------------------------------
        dx = self._cell_x_max[cell_idx] - self._cell_x_min[cell_idx]
        dy = self._cell_y_max[cell_idx] - self._cell_y_min[cell_idx]
        dz = self._cell_z_max[cell_idx] - self._cell_z_min[cell_idx]

        if dx <= 1e-15 or dy <= 1e-15 or dz <= 1e-15:
            # Degenerate cell — fall back to inverse-distance weighting.
            return self._idw_fallback(point, values)

        xi = np.clip(2.0 * (px - self._cell_x_min[cell_idx]) / dx - 1.0, -1.0, 1.0)
        eta = np.clip(2.0 * (py - self._cell_y_min[cell_idx]) / dy - 1.0, -1.0, 1.0)
        zeta = np.clip(2.0 * (pz - self._cell_z_min[cell_idx]) / dz - 1.0, -1.0, 1.0)

        # ------------------------------------------------------------------
        # 4. Evaluate 1D GLL Lagrange polynomials at (xi, eta, zeta).
        # ------------------------------------------------------------------
        ell_xi = self._lagrange_basis(xi)  # [ngll]
        ell_eta = self._lagrange_basis(eta)  # [ngll]
        ell_zeta = self._lagrange_basis(zeta)  # [ngll]

        # 3D tensor-product weights: w_{ijk} = ell_xi[i] * ell_eta[j] * ell_zeta[k]
        # Shape: [125]
        weights = np.outer(np.outer(ell_xi, ell_eta).ravel(), ell_zeta).ravel()
        weights /= weights.sum()  # partition of unity

        # ------------------------------------------------------------------
        # 5. Weighted sum over the cell's 125 GLL nodes.
        # ------------------------------------------------------------------
        node_indices = self._cell_gll_node_index[cell_idx]  # [125]
        cell_values = values[node_indices]  # [125, ...]
        return np.tensordot(weights, cell_values, axes=(0, 0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lagrange_basis(self, xi: float) -> np.ndarray:
        """Evaluate all 1D GLL Lagrange polynomials at reference coordinate *xi*.

        L_i(xi) = prod_{j != i} (xi - xi_j) / (xi_i - xi_j)

        Returns shape ``(ngll,)``.
        """
        n = self._ngll
        ell = np.ones(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ell[i] *= (xi - _GLL_XI_1D[j]) / (_GLL_XI_1D[i] - _GLL_XI_1D[j])
        return ell

    def _idw_fallback(self, point: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Inverse-distance-weighted average (degenerate-cell fallback)."""
        distances, indices = self._node_tree.query(point, k=min(125, len(self._gll_node_coords)))
        if isinstance(indices, np.integer):
            indices = np.array([indices])
            distances = np.array([distances])
        weights = 1.0 / (distances + 1e-300)
        weights /= weights.sum()
        cell_values = values[indices]
        return np.tensordot(weights, cell_values, axes=(0, 0))


def reconstruct_cell_gll_index(
    gll_node_coords: np.ndarray,
    ngll: int = 5,
) -> np.ndarray:
    """Reconstruct cell_gll_node_index from GLL node coordinates.

    For a rectilinear Cartesian tile, GLL nodes form a structured grid.
    This function extracts the unique axis positions and reconstructs
    which 125 GLL nodes belong to each cell.

    Args:
        gll_node_coords: Shape ``(n_unique_gll, 3)``, float64.
        ngll: Number of GLL nodes per dimension (default 5).

    Returns:
        Shape ``(n_cell, ngll**3)`` int64 array, or ``None`` if
        reconstruction fails (non-rectilinear mesh).
    """
    coords = np.asarray(gll_node_coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return None

    # Snap coordinates to micron grid to eliminate float64 rounding noise
    # (observed ~2e-13 differences across GLL nodes at same physical position)
    coords_snapped = np.round(coords, decimals=6)
    x_unique = np.sort(np.unique(coords_snapped[:, 0]))
    y_unique = np.sort(np.unique(coords_snapped[:, 1]))
    z_unique = np.sort(np.unique(coords_snapped[:, 2]))

    step = ngll - 1  # nodes per cell edge minus 1 (4 for NGLL=5)
    nx_cells = (len(x_unique) - 1) // step
    ny_cells = (len(y_unique) - 1) // step
    nz_cells = (len(z_unique) - 1) // step

    if nx_cells <= 0 or ny_cells <= 0 or nz_cells <= 0:
        return None
    if (len(x_unique) - 1) % step != 0 or (len(y_unique) - 1) % step != 0 or (len(z_unique) - 1) % step != 0:
        return None  # non-uniform grid

    # Build (x_idx, y_idx, z_idx) → flat node index mapping
    # Use a dict for sparse lookup
    coord_to_idx = {}
    for i in range(coords.shape[0]):
        xi = int(np.searchsorted(x_unique, coords_snapped[i, 0]))
        yi = int(np.searchsorted(y_unique, coords_snapped[i, 1]))
        zi = int(np.searchsorted(z_unique, coords_snapped[i, 2]))
        coord_to_idx[(xi, yi, zi)] = i

    n_cells = nx_cells * ny_cells * nz_cells
    cell_gll_index = np.full((n_cells, ngll**3), -1, dtype=np.int64)
    ci = 0
    for cx in range(nx_cells):
        x_start = cx * step
        for cy in range(ny_cells):
            y_start = cy * step
            for cz in range(nz_cells):
                z_start = cz * step
                for i in range(ngll):
                    for j in range(ngll):
                        for k in range(ngll):
                            gxi = x_start + i
                            gyi = y_start + j
                            gzi = z_start + k
                            node_idx = coord_to_idx.get((gxi, gyi, gzi), -1)
                            local_idx = i * ngll * ngll + j * ngll + k
                            cell_gll_index[ci, local_idx] = node_idx
                ci += 1

    # Validate: all entries must be >= 0
    if np.any(cell_gll_index < 0):
        return None

    return cell_gll_index
