"""GLL interpolation of strain at arbitrary spatial positions."""

import numpy as np
import numpy.typing as npt

from gf_post.geometry import gll_nodes_1d, lagrange_basis_3d


def interpolate_strain(
    strain_at_element: npt.NDArray[np.float64], xi: float, eta: float, zeta: float
) -> npt.NDArray[np.float64]:
    """Interpolate strain from GLL nodes to an arbitrary point.

    Computes ε(ξ,η,ζ) = Σ_ijk l_i(ξ)·l_j(η)·l_k(ζ)·ε_ijk
    for each of the 6 Voigt strain components.

    Args:
        strain_at_element: [NGLL, NGLL, NGLL, 6] strain at all GLL nodes.
        xi: natural coordinate in [-1, 1].
        eta: natural coordinate in [-1, 1].
        zeta: natural coordinate in [-1, 1].

    Returns:
        [6] interpolated strain components [εxx, εyy, εzz, εxy, εxz, εyz].
    """
    ngll = strain_at_element.shape[0]
    nodes_1d = gll_nodes_1d(ngll - 1)
    basis = lagrange_basis_3d((xi, eta, zeta), nodes_1d)  # [ngll, ngll, ngll]

    result = np.empty(6, dtype=np.float64)
    for c in range(6):
        result[c] = np.sum(basis * strain_at_element[:, :, :, c])

    return result
