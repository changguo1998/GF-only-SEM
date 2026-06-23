"""Source locator — find containing elements and compute Lagrange weights.

Given a source position (x_s, y_s, z_s) on the free surface, locate all
free-surface elements that contain the source via Newton iteration in
natural coordinates, then compute Lagrange interpolation weights.
The source may lie on a shared face, edge, or vertex — all containing
elements are returned with normalized weights.

These precomputed weights and element list are stored in config.h5 so
the forward solver can inject the source without runtime search.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from preprocess.topology_reader import TopologyData
from preprocess.gll_geometry import gll_quadrature_points

# ---------------------------------------------------------------------------
# Newton iteration for point-in-hexahedron
# ---------------------------------------------------------------------------

# Linear shape functions for 8-node hex (same as gll_geometry._linear_shape_derivs)
HEX_REF_CORNERS_LOC = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
], dtype=np.float64)


def _linear_shape_derivs_loc(
    xi: float, eta: float, zeta: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Linear hex shape functions and their derivs at (xi, eta, zeta).

    Returns (N_vals [8], dN [8,3]) where dN[a,m] = ∂N_a/∂ξ_m.
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
    """Newton iteration to find natural coords (ξ,η,ζ) of a physical point.

    Uses the linear hex mapping: x(ξ,η,ζ) = Σ N_a(ξ,η,ζ) · x_a.

    Args:
        x_target: [3] physical point.
        corners:  [8, 3] physical corner coordinates in GMSH order.
        max_iter: Maximum Newton iterations.
        tol:      Convergence tolerance on |Δξ|.

    Returns:
        [3] (ξ, η, ζ) in reference domain, or None if not converged.
    """
    xi = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # start at element center

    for _ in range(max_iter):
        _N, dN = _linear_shape_derivs_loc(xi[0], xi[1], xi[2])
        x_current = _N @ corners                         # [3] current physical estimate
        J = dN.T @ corners                               # [3, 3] dx/dξ

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

    # Check that the result is inside (or very close to) [-1, 1]³
    margin = 1e-8
    if np.any(xi < -(1.0 + margin)) or np.any(xi > 1.0 + margin):
        return None

    return xi


# ---------------------------------------------------------------------------
# Lagrange interpolation weights at natural coordinates
# ---------------------------------------------------------------------------


def _lagrange_basis_at(
    xi_eval: float, nodes: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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
    xi_vec: npt.NDArray[np.float64],
    gll_pts: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute 3-D Lagrange interpolation weights at natural coords.

    w_ijk = l_i(ξ_s) · l_j(η_s) · l_k(ζ_s)

    Args:
        xi_vec: [3] natural coords (ξ, η, ζ).
        gll_pts: [NGLL] 1-D GLL nodal points.

    Returns:
        weights: [NGLL, NGLL, NGLL] Lagrange interpolation weights.
    """
    NGLL = len(gll_pts)
    lx = _lagrange_basis_at(xi_vec[0], gll_pts)
    ly = _lagrange_basis_at(xi_vec[1], gll_pts)
    lz = _lagrange_basis_at(xi_vec[2], gll_pts)
    return lx[:, np.newaxis, np.newaxis] * ly[np.newaxis, :, np.newaxis] * lz[np.newaxis, np.newaxis, :]


# ---------------------------------------------------------------------------
# Source location function for free surface elements
# ---------------------------------------------------------------------------


def locate_source(
    topology: TopologyData,
    source_xyz: npt.NDArray[np.float64],
    gll_coords: npt.NDArray[np.float64],
    boundary_tag: npt.NDArray[np.int64],
    N: int,
) -> dict:
    """Locate source on free surface, return element list + Lagrange weights.

    Searches only free-surface elements (boundary_tag == 1).
    Newton iteration in natural coords for each candidate element.
    Computes Lagrange interpolation weights at the source position.
    Weights are normalized so Σ w_ijk = 1 across all containing elements.

    Args:
        topology:    Mesh topology data.
        source_xyz:  [3] source position (x, y, z) — z should be on free surface.
        gll_coords:  [n_cell, NGLL, NGLL, NGLL, 3] GLL node positions.
        boundary_tag: [n_surface] surface tags (0=interior, 1=free, 2=absorbing).
        N:           Polynomial order.

    Returns:
        dict with:
          element_ids: [n_src_elem] 1-based global element IDs.
          xi:          [n_src_elem] ξ natural coordinate.
          eta:         [n_src_elem] η natural coordinate.
          zeta:        [n_src_elem] ζ natural coordinate.
          weights:     [n_src_elem, NGLL, NGLL, NGLL] Lagrange w_ijk (normalized).
          n_src_elem:  int count.
    """
    n_cell = topology.n_cell
    c2s = topology.cell_to_surface
    NGLL = N + 1
    gll_pts = gll_quadrature_points(N)

    # Identify free-surface elements (those with at least one face tagged 1)
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
            "Check that the mesh has at least one surface with z ≈ z_min."
        )

    # Get corner coordinates for each candidate element via the gll_coords
    # The 8 corners of an element are at GLL indices (0,0,0), (NGLL-1,0,0), ...
    # GMSH ordering from gmsh_to_hdf5 + _get_cell_vertex_ids:
    corner_indices = [
        (0, 0, 0),                                           # v0: -1,-1,-1
        (NGLL - 1, 0, 0),                                    # v1: +1,-1,-1
        (NGLL - 1, NGLL - 1, 0),                             # v2: +1,+1,-1
        (0, NGLL - 1, 0),                                    # v3: -1,+1,-1
        (0, 0, NGLL - 1),                                    # v4: -1,-1,+1
        (NGLL - 1, 0, NGLL - 1),                             # v5: +1,-1,+1
        (NGLL - 1, NGLL - 1, NGLL - 1),                      # v6: +1,+1,+1
        (0, NGLL - 1, NGLL - 1),                             # v7: -1,+1,+1
    ]

    element_ids: list[int] = []
    xi_list: list[float] = []
    eta_list: list[float] = []
    zeta_list: list[float] = []
    weights_list: list[npt.NDArray[np.float64]] = []

    for e in free_surface_cells:
        corners = np.array([
            gll_coords[e, ci[0], ci[1], ci[2]] for ci in corner_indices
        ])

        xi_vec = _newton_find_xi(source_xyz, corners)
        if xi_vec is None:
            continue

        w = compute_source_weights(xi_vec, gll_pts)

        # Check that weights sum is reasonable (should be ~1 for a single elem)
        w_sum = np.sum(w)
        if w_sum < 1e-12:
            continue

        element_ids.append(e + 1)   # 1-based global element ID
        xi_list.append(float(xi_vec[0]))
        eta_list.append(float(xi_vec[1]))
        zeta_list.append(float(xi_vec[2]))
        weights_list.append(w)

    if not element_ids:
        raise ValueError(
            f"Source at {source_xyz} not contained in any free-surface element. "
            f"Checked {len(free_surface_cells)} candidate(s)."
        )

    # Normalize weights across all containing elements
    total_weight = float(sum(np.sum(w) for w in weights_list))
    weights_norm = [w / total_weight for w in weights_list]

    return {
        "element_ids": np.array(element_ids, dtype=np.int64),
        "xi": np.array(xi_list, dtype=np.float64),
        "eta": np.array(eta_list, dtype=np.float64),
        "zeta": np.array(zeta_list, dtype=np.float64),
        "weights": np.array(weights_norm, dtype=np.float64),
        "n_src_elem": len(element_ids),
    }