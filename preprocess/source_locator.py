"""Source locator -- find containing elements and compute Lagrange weights.

Given a source position (x_s, y_s, z_s), locate all elements that contain
the source via Newton iteration in natural coordinates, then compute
Lagrange interpolation weights.

Two modes:
- **Surface mode** (default): source_z == z_min, searches only free-surface
  elements (boundary_tag == 1). The source may lie on a shared face, edge,
  or vertex -- all containing elements are returned with normalized weights.
- **Buried mode** (is_pml provided, source_z != z_min): searches all non-PML
  elements via AABB bounding box test + Newton iteration. Source should be
  inside a single element; >1 elements triggers a warning.

These precomputed weights and element list are stored in config.h5 so
the forward solver can inject the source without runtime search.
"""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt

from preprocess.gll_geometry import gll_quadrature_points
from preprocess.topology_reader import TopologyData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Newton iteration for point-in-hexahedron
# ---------------------------------------------------------------------------

# Linear shape functions for 8-node hex (same as gll_geometry._linear_shape_derivs)
HEX_REF_CORNERS_LOC = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=np.float64,
)


def _linear_shape_derivs_loc(
    xi: float, eta: float, zeta: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Linear hex shape functions and their derivs at (xi, eta, zeta).

    Returns (N_vals [8], dN [8,3]) where dN[a,m] = dN_a/dxi_m.
    """
    N_vals = np.zeros(8, dtype=np.float64)
    dN = np.zeros((8, 3), dtype=np.float64)
    for a in range(8):
        ca, cb, cc = HEX_REF_CORNERS_LOC[a]
        t0 = 0.125
        N_vals[a] = t0 * (1 + ca * xi) * (1 + cb * eta) * (1 + cc * zeta)
        dN[a, 0] = t0 * ca * (1 + cb * eta) * (1 + cc * zeta)
        dN[a, 1] = t0 * (1 + ca * xi) * cb * (1 + cc * zeta)
        dN[a, 2] = t0 * (1 + ca * xi) * (1 + cb * eta) * cc
    return N_vals, dN


def _newton_find_xi(
    x_target: npt.NDArray[np.float64],
    corners: npt.NDArray[np.float64],
    max_iter: int = 50,
    tol: float = 1e-12,
) -> npt.NDArray[np.float64] | None:
    """Newton iteration to find natural coords (xi,eta,zeta) of a physical point.

    Uses the linear hex mapping: x(xi,eta,zeta) = sum N_a(xi,eta,zeta) * x_a.

    Args:
        x_target: [3] physical point.
        corners:  [8, 3] physical corner coordinates in GMSH order.
        max_iter: Maximum Newton iterations.
        tol:      Convergence tolerance on |Deltaxi|.

    Returns:
        [3] (xi, eta, zeta) in reference domain, or None if not converged.
    """
    xi = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # start at element center

    for _ in range(max_iter):
        _N, dN = _linear_shape_derivs_loc(xi[0], xi[1], xi[2])
        x_current = _N @ corners  # [3] current physical estimate
        J = dN.T @ corners  # [3, 3] dx/dxi

        residual = x_target - x_current
        try:
            dxi = np.linalg.solve(J, residual)
        except np.linalg.LinAlgError:
            return None

        xi = xi + dxi
        if np.linalg.norm(dxi) < tol:
            break
    else:
        # Did not converge within max_iter
        return None

    # Check that the result is inside (or very close to) [-1, 1]^3
    margin = 1e-8
    if np.any(xi < -(1.0 + margin)) or np.any(xi > 1.0 + margin):
        return None

    return xi


# ---------------------------------------------------------------------------
# Lagrange interpolation weights at natural coordinates
# ---------------------------------------------------------------------------


def _lagrange_basis_at(xi_eval: float, nodes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate 1-D Lagrange basis on given nodal points at xi_eval.

    Args:
        xi_eval: Evaluation point in [-1, 1].
        nodes:   Nodal positions [NGLL] (e.g., GLL quadrature points).

    Returns:
        w_i: [NGLL] Lagrange basis values l_i(xi_eval).
    """
    n = len(nodes)
    w = np.ones(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] *= (xi_eval - nodes[j]) / (nodes[i] - nodes[j])
    return w


def compute_source_weights(
    xi_vec: npt.NDArray[np.float64], gll_pts: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute 3-D Lagrange interpolation weights at natural coords.

    w_ijk = l_i(xi_s) * l_j(eta_s) * l_k(zeta_s)

    Args:
        xi_vec: [3] natural coords (xi, eta, zeta).
        gll_pts: [NGLL] 1-D GLL nodal points.

    Returns:
        weights: [NGLL, NGLL, NGLL] Lagrange interpolation weights.
    """
    NGLL = len(gll_pts)
    lx = _lagrange_basis_at(xi_vec[0], gll_pts)
    ly = _lagrange_basis_at(xi_vec[1], gll_pts)
    lz = _lagrange_basis_at(xi_vec[2], gll_pts)
    return (
        lx[:, np.newaxis, np.newaxis]
        * ly[np.newaxis, :, np.newaxis]
        * lz[np.newaxis, np.newaxis, :]
    )


# ---------------------------------------------------------------------------
# Bounding box helper for buried source candidate search
# ---------------------------------------------------------------------------


def _find_candidate_elements(
    source_xyz: npt.NDArray[np.float64],
    gll_coords: npt.NDArray[np.float64],
    is_pml: npt.NDArray[np.bool_] | None = None,
) -> list[int]:
    """Find elements whose AABB contains source_xyz.

    Uses the GLL node bounding box (min/max over all GLL nodes) as a fast
    rejection filter before Newton iteration.

    Args:
        source_xyz:  [3] source position.
        gll_coords:  [n_cell, NGLL, NGLL, NGLL, 3] GLL node positions.
        is_pml:      [n_cell] bool mask, True for PML elements to exclude.

    Returns:
        List of element indices (0-based) whose AABB contains source_xyz.
    """
    n_cell = gll_coords.shape[0]
    candidates: list[int] = []

    # Compute per-element AABB once
    # gll_coords shape: [n_cell, NGLL, NGLL, NGLL, 3]
    x_min = gll_coords[..., 0].min(axis=(1, 2, 3))
    x_max = gll_coords[..., 0].max(axis=(1, 2, 3))
    y_min = gll_coords[..., 1].min(axis=(1, 2, 3))
    y_max = gll_coords[..., 1].max(axis=(1, 2, 3))
    z_min = gll_coords[..., 2].min(axis=(1, 2, 3))
    z_max = gll_coords[..., 2].max(axis=(1, 2, 3))

    margin = 1e-10
    sx, sy, sz = float(source_xyz[0]), float(source_xyz[1]), float(source_xyz[2])

    for e in range(n_cell):
        if is_pml is not None and bool(is_pml[e]):
            continue
        if (
            x_min[e] - margin <= sx <= x_max[e] + margin
            and y_min[e] - margin <= sy <= y_max[e] + margin
            and z_min[e] - margin <= sz <= z_max[e] + margin
        ):
            candidates.append(e)

    return candidates


# ---------------------------------------------------------------------------
# Corner extraction for Newton iteration (GMSH hex ordering)
# ---------------------------------------------------------------------------


def _get_element_corners(
    gll_coords: npt.NDArray[np.float64], e: int, ngll: int
) -> npt.NDArray[np.float64]:
    """Extract the 8 corner vertices of element e from GLL coords."""
    idx = ngll - 1
    corners = [
        gll_coords[e, 0, 0, 0],
        gll_coords[e, idx, 0, 0],
        gll_coords[e, idx, idx, 0],
        gll_coords[e, 0, idx, 0],
        gll_coords[e, 0, 0, idx],
        gll_coords[e, idx, 0, idx],
        gll_coords[e, idx, idx, idx],
        gll_coords[e, 0, idx, idx],
    ]
    return np.array(corners, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main entry point: locate source (surface or buried)
# ---------------------------------------------------------------------------


def locate_source(
    topology: TopologyData,
    source_xyz: npt.NDArray[np.float64],
    gll_coords: npt.NDArray[np.float64],
    boundary_tag: npt.NDArray[np.int64],
    N: int,
    is_pml: npt.NDArray[np.bool_] | None = None,
) -> dict:
    """Locate source position, return element list + Lagrange weights.

    Two modes determined automatically:
    - **Surface mode**: when is_pml is None (legacy call) or source_z is
      the domain's z_min. Searches only free-surface elements (boundary_tag==1).
    - **Buried mode**: when is_pml is provided and source_z != z_min. Searches
      all non-PML elements via AABB bounding box + Newton iteration.

    Args:
        topology:    Mesh topology data.
        source_xyz:  [3] source position (x, y, z).
        gll_coords:  [n_cell, NGLL, NGLL, NGLL, 3] GLL node positions.
        boundary_tag: [n_surface] surface tags (0=interior, 1=free, 2=absorbing).
        N:           Polynomial order.
        is_pml:      [n_cell] bool mask, True for PML elements. Required for
                     buried mode; ignored in surface mode.

    Returns:
        dict with:
          element_ids: [n_src_elem] 0-based element indices (match partition local_element_ids).
          xi:          [n_src_elem] xi natural coordinate.
          eta:         [n_src_elem] eta natural coordinate.
          zeta:        [n_src_elem] zeta natural coordinate.
          weights:     [n_src_elem, NGLL, NGLL, NGLL] Lagrange w_ijk (normalized).
          n_src_elem:  int count.
          mode:        str, "surface" or "buried".
    """
    n_cell = topology.n_cell
    c2s = topology.cell_to_surface
    NGLL = N + 1
    gll_pts = gll_quadrature_points(N)

    # Determine mode
    # Surface mode: is_pml is None (legacy) OR source_z matches z_min
    zmin = float(gll_coords[:, :, :, :, 2].min())
    is_buried = is_pml is not None and abs(float(source_xyz[2]) - zmin) > 1e-8 * abs(zmin or 1.0)

    if is_buried:
        # ── Buried mode: search all non-PML elements ──
        if is_pml is None:
            raise ValueError("Buried mode requires is_pml array.")
        logger.debug(
            "Buried source at z=%.4f m, searching non-PML elements...", float(source_xyz[2])
        )
        candidates = _find_candidate_elements(source_xyz, gll_coords, is_pml)
        if not candidates:
            raise ValueError(
                f"Buried source at {source_xyz} not contained in any non-PML element."
            )
        element_iter = candidates
    else:
        # ── Surface mode: search free-surface elements ──
        logger.debug("Surface source at z=%.4f m, searching free-surface elements...", zmin)
        free_surface_cells: list[int] = []
        for e in range(n_cell):
            for fi in range(6):
                sid = abs(int(c2s[e, fi])) - 1
                if boundary_tag[sid] == 1:
                    free_surface_cells.append(e)
                    break

        if not free_surface_cells:
            raise ValueError(
                "No free-surface elements found. "
                "Check that the mesh has at least one surface with z asymp z_min."
            )
        element_iter = free_surface_cells

    # Common: Newton iteration and weight computation for each candidate
    element_ids: list[int] = []
    xi_list: list[float] = []
    eta_list: list[float] = []
    zeta_list: list[float] = []
    weights_list: list[npt.NDArray[np.float64]] = []

    for e in element_iter:
        corners = _get_element_corners(gll_coords, e, NGLL)

        xi_vec = _newton_find_xi(source_xyz, corners)
        if xi_vec is None:
            continue

        w = compute_source_weights(xi_vec, gll_pts)

        # Check that weights sum is reasonable (should be ~1 for a single elem)
        w_sum = np.sum(w)
        if w_sum < 1e-12:
            continue

        element_ids.append(e)  # 0-based element index (matches partition local_element_ids)
        xi_list.append(float(xi_vec[0]))
        eta_list.append(float(xi_vec[1]))
        zeta_list.append(float(xi_vec[2]))
        weights_list.append(w)

    if not element_ids:
        raise ValueError(
            f"Source at {source_xyz} not contained in any candidate element. "
            f"Checked {len(element_iter)} candidate(s)."
        )

    if is_buried and len(element_ids) > 1:
        logger.warning(
            "Buried source at %s found in %d element(s) — may lie on element "
            "face/edge. Verify source position is correct.",
            source_xyz,
            len(element_ids),
        )

    # Normalize weights across all containing elements
    # (mathematically required: scatter_to_rank is additive, so shared nodes
    #  would get Nx amplitude without normalization)
    total_weight = float(sum(np.sum(w) for w in weights_list))
    weights_norm = [w / total_weight for w in weights_list]

    return {
        "element_ids": np.array(element_ids, dtype=np.int64),
        "xi": np.array(xi_list, dtype=np.float64),
        "eta": np.array(eta_list, dtype=np.float64),
        "zeta": np.array(zeta_list, dtype=np.float64),
        "weights": np.array(weights_norm, dtype=np.float64),
        "n_src_elem": len(element_ids),
        "mode": "buried" if is_buried else "surface",
    }
