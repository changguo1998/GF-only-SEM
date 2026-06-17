"""Point-in-hexahedron search using Newton iteration.

Given a spatial point and candidate element IDs, runs Newton's method in
natural coordinates (ξ, η, ζ) to find the containing element and the
natural coordinates of the point within it.
"""

import numpy as np
import numpy.typing as npt

from gf_post.geometry import gll_nodes_1d, lagrange_basis_3d


def find_containing_element(
    point: npt.NDArray[np.float64],
    candidates: npt.NDArray[np.int64],
    gll_coords: npt.NDArray[np.float64],
    dxi_dx: npt.NDArray[np.float64],
    tol: float = 1e-10,
    max_iter: int = 50,
) -> tuple[int, float, float, float]:
    """Find the element containing a point via Newton iteration.
    
    For each candidate element, evaluate the position using GLL interpolation
    and iterate in natural coordinates (ξ, η, ζ) until convergence.
    
    Args:
        point: (3,) spatial coordinates [x, y, z].
        candidates: 1-based global element IDs to search.
        gll_coords: [n_cell, NGLL, NGLL, NGLL, 3] GLL node coordinates.
        dxi_dx: [n_cell, NGLL, NGLL, NGLL, 3, 3] Jacobian inverse.
        tol: convergence tolerance on coordinate residual.
        max_iter: maximum Newton iterations.
    
    Returns:
        (element_id, xi, eta, zeta) — 1-based element ID and natural coords.
    
    Raises:
        ValueError: if no containing element found among candidates.
    """
    for eid_1based in candidates:
        idx = int(eid_1based) - 1
        result = _newton_iterate(point, idx, gll_coords, dxi_dx, tol, max_iter)
        if result is not None:
            xi, eta, zeta = result
            # Verify the natural coords are in [-1, 1]
            if -1 - tol <= xi <= 1 + tol and -1 - tol <= eta <= 1 + tol and -1 - tol <= zeta <= 1 + tol:
                return int(eid_1based), float(xi), float(eta), float(zeta)
    
    raise ValueError(
        f"Point {tuple(point)} not found in any of {len(candidates)} candidate elements"
    )


def _newton_iterate(
    point: npt.NDArray[np.float64],
    idx: int,
    gll_coords: npt.NDArray[np.float64],
    dxi_dx: npt.NDArray[np.float64],
    tol: float,
    max_iter: int,
) -> tuple[float, float, float] | None:
    """Newton iteration for a single element.
    
    Given physical point, find natural coordinates (ξ, η, ζ) such that
    x(ξ,η,ζ) = point. Uses precomputed dξ/dx as the Jacobian.
    
    Returns (xi, eta, zeta) or None if not converging.
    """
    ngll = gll_coords.shape[1]
    nodes_1d = gll_nodes_1d(ngll - 1)
    
    # Initialize with centroid
    xi, eta, zeta = 0.0, 0.0, 0.0
    
    for _ in range(max_iter):
        # Compute position at current (xi, eta, zeta) using GLL interpolation
        coords_elem = gll_coords[idx]  # [ngll, ngll, ngll, 3]
        
        # Interpolate position
        basis = lagrange_basis_3d((xi, eta, zeta), nodes_1d)
        x_pos = np.sum(basis[:, :, :, np.newaxis] * coords_elem, axis=(0, 1, 2))
        
        # Residual
        r = x_pos - point
        if np.linalg.norm(r) < tol:
            return float(xi), float(eta), float(zeta)
        
        # Jacobian: dx/dξ = (dξ/dx)^{-1}
        # We have dxi_dx at all GLL nodes — interpolate at current (xi, eta, zeta)
        jacobian_inv = np.sum(basis[:, :, :, np.newaxis, np.newaxis] * dxi_dx[idx],
                              axis=(0, 1, 2))  # [3, 3]
        
        # Newton step in natural coordinates: δξ = (dξ/dx) @ r
        delta = jacobian_inv @ r
        xi -= delta[0]
        eta -= delta[1]
        zeta -= delta[2]
    
    return None