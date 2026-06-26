"""Spatial index over hexahedral element centroids.

Uses scipy KD-tree for nearest-centroid queries.
"""

import numpy as np
from scipy.spatial import KDTree


class ElementIndex:
    """KD-tree index over non-PML element centroids."""

    def __init__(self, gll_coords: np.ndarray, is_pml: np.ndarray | None = None):
        """Build KD-tree on element centroids.

        Args:
            gll_coords: shape [n_cell, NGLL, NGLL, NGLL, 3] — GLL node coordinates.
            is_pml: shape [n_cell] — optional PML mask. If None, all elements are indexed.
        """
        # Centroid = mean of all GLL nodes
        centroids = gll_coords.mean(axis=(1, 2, 3))  # [n_cell, 3]

        if is_pml is not None:
            # Exclude PML elements
            mask = is_pml == 0
            self._all_indices = np.arange(len(centroids))
            self._indices = self._all_indices[mask]
            centroids = centroids[mask]
        else:
            self._indices = np.arange(len(centroids))

        self._tree = KDTree(centroids)
        self._n_elements = len(centroids)

    def query(self, point: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest element centroids.

        Args:
            point: shape (3,) or (m, 3) query coordinates.
            k: number of nearest neighbors.

        Returns:
            (indices, distances) where indices map to original element IDs.
        """
        point = np.asarray(point)
        if point.ndim == 1:
            point = point[np.newaxis, :]

        dist, local_idx = self._tree.query(point, k=k)
        global_idx = self._indices[local_idx]
        return global_idx, dist
