"""Model loader — placeholder that returns constant material values.

In the current milestone, this module provides a stand-in for a future
3D model interpolation routine. It accepts a model path and GLL node
coordinates and returns uniform default values (vp=3000, vs=1500,
density=2500) at each GLL node.

Future work will replace this with actual 3D model file I/O and
interpolation (e.g., NetCDF, HDF5, or user-provided grid data).
"""

import numpy as np
import numpy.typing as npt


def load_and_interpolate(
    model_path: str | None,
    gll_coords: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load and interpolate material model at GLL node positions.

    Placeholder: ignores model_path and returns constant values.

    Args:
        model_path: Path to material model file (ignored in placeholder).
        gll_coords: GLL node coordinates, shape [n_cell, NGLL, NGLL, NGLL, 3].

    Returns:
        Tuple of (vp, vs, density), each [n_cell, NGLL, NGLL, NGLL] float64.
    """
    shape = gll_coords.shape[:-1]  # (n_cell, NGLL, NGLL, NGLL)
    vp = np.full(shape, 3000.0, dtype=np.float64)
    vs = np.full(shape, 1500.0, dtype=np.float64)
    density = np.full(shape, 2500.0, dtype=np.float64)
    return vp, vs, density