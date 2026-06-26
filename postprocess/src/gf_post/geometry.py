"""GLL geometry utilities for SEM postprocessing.

Computes 1D/3D GLL nodes, quadrature weights, and Lagrange basis functions
used in the strain interpolation pipeline.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import legendre


def gll_nodes_1d(N: int) -> npt.NDArray[np.float64]:
    """Compute 1D GLL quadrature points in [-1, 1] for polynomial order N.

    Returns N+1 points including the endpoints ±1.

    Args:
        N: Polynomial order (number of interior nodes = N-1).

    Returns:
        1D array of shape (N+1,) sorted ascending.
    """
    if N == 0:
        return np.array([0.0], dtype=np.float64)
    if N == 1:
        return np.array([-1.0, 1.0], dtype=np.float64)

    # Roots of derivative of Legendre polynomial P_N
    dpoly = np.polyder(legendre(N))
    xi_roots = np.sort(np.roots(dpoly)).real
    xi_roots = xi_roots[(xi_roots > -1 + 1e-12) & (xi_roots < 1 - 1e-12)]
    points = np.concatenate([[-1.0], xi_roots, [1.0]])
    return np.ascontiguousarray(points, dtype=np.float64)


def gll_weights_1d(N: int) -> npt.NDArray[np.float64]:
    """Compute 1D GLL quadrature weights.

    The weights sum to 2 (the interval length [-1, 1]).

    Args:
        N: Polynomial order.

    Returns:
        1D array of shape (N+1,).
    """
    nodes = gll_nodes_1d(N)
    n = N + 1
    w = np.empty(n, dtype=np.float64)
    for i in range(n):
        pn = legendre(N)(nodes[i])
        w[i] = 2.0 / (N * (N + 1) * pn * pn)
    return w


def gll_nodes_3d(N: int) -> npt.NDArray[np.float64]:
    """Compute 3D tensor-product GLL nodes in [-1, 1]^3.

    Args:
        N: Polynomial order.

    Returns:
        Array of shape (N+1, N+1, N+1, 3) with (x, y, z) at each node.
        Indexing convention: nodes[k, j, i, :] = (xi_i, xi_j, xi_k).
    """
    nodes_1d = gll_nodes_1d(N)
    ngll = len(nodes_1d)
    # Build 3D grid: k (z/ζ), j (y/η), i (x/ξ)
    X, Y, Z = np.meshgrid(nodes_1d, nodes_1d, nodes_1d, indexing="ij")
    # Result: [k,j,i,3] but meshgrid gives [i,j,k] — transpose to [k,j,i]
    return np.stack(
        [X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)], axis=-1
    ).transpose(2, 1, 0, 3)


def lagrange_basis_1d(xi: float, nodes: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Evaluate all 1D Lagrange basis polynomials at a point xi.

    L_i(xi) = Π_{j≠i} (xi - x_j) / (x_i - x_j)

    Args:
        xi: Evaluation point in [-1, 1].
        nodes: GLL nodes of shape (ngll,).

    Returns:
        Array of shape (ngll,) with L_0(xi), ..., L_{N}(xi).
    """
    ngll = len(nodes)
    basis = np.empty(ngll, dtype=np.float64)
    for i in range(ngll):
        li = 1.0
        for j in range(ngll):
            if j != i:
                li *= (xi - nodes[j]) / (nodes[i] - nodes[j])
        basis[i] = li
    return basis


def lagrange_basis_3d(
    point: tuple[float, float, float], nodes_1d: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Evaluate 3D tensor-product Lagrange basis at a point.

    L_ijk(ξ, η, ζ) = L_i(ξ) · L_j(η) · L_k(ζ)

    Args:
        point: (xi, eta, zeta) in [-1, 1]^3.
        nodes_1d: 1D GLL nodes of shape (ngll,).

    Returns:
        Array of shape (ngll, ngll, ngll) with 3D basis values.
        Indexing: basis[k, j, i] = L_i(xi)·L_j(eta)·L_k(zeta).
    """
    xi, eta, zeta = point
    l_xi = lagrange_basis_1d(xi, nodes_1d)  # shape (ngll,)
    l_eta = lagrange_basis_1d(eta, nodes_1d)  # shape (ngll,)
    l_zeta = lagrange_basis_1d(zeta, nodes_1d)  # shape (ngll,)
    # Tensor product: [k,j,i] = l_zeta[k] * l_eta[j] * l_xi[i]
    return l_zeta[:, np.newaxis, np.newaxis] * l_eta[:, np.newaxis] * l_xi
