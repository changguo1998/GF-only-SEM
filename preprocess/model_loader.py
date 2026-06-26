"""Model loader — evaluates config material functions at GLL nodes.

Reads Vp, Vs, and density from the configuration's material callables
(vp_m_s, vs_m_s, density_kg_m3) at every GLL node coordinate.
Falls back to hardcoded defaults only if no material functions exist.
"""

import types
from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def load_and_interpolate(
    model_path: str | None,
    gll_coords: npt.NDArray[np.float64],
    config: types.ModuleType | None = None,
    *,
    vp_func: Callable[[float, float, float], float] | None = None,
    vs_func: Callable[[float, float, float], float] | None = None,
    density_func: Callable[[float, float, float], float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate material model at GLL node positions.

    Material properties are obtained from config-provided callables
    ``vp_m_s(x, y, z)``, ``vs_m_s(x, y, z)``, ``density_kg_m3(x, y, z)``.
    If neither the config module nor explicit functions are given,
    constant defaults are used (backward compatibility).

    Args:
        model_path: Ignored — material is defined purely in config.
        gll_coords: GLL node coordinates, shape [n_cell, NGLL, NGLL, NGLL, 3].
        config: Config module (e.g. from load_config / config.py import).
            When provided, its ``vp_m_s``, ``vs_m_s``, ``density_kg_m3``
            attributes are used if they exist.
        vp_func: Explicit P-wave velocity callable.  Takes precedence over
            ``config.vp_m_s``.
        vs_func: Explicit S-wave velocity callable.
        density_func: Explicit density callable.

    Returns:
        Tuple of (vp, vs, density), each [n_cell, NGLL, NGLL, NGLL] float64.
    """
    shape = gll_coords.shape[:-1]  # (n_cell, NGLL, NGLL, NGLL)
    flat = gll_coords.reshape(-1, 3)  # (n_points, 3)

    # Resolve material callables — explicit args > config attributes > defaults
    vp_func = vp_func or _resolve_callable(config, "vp_m_s", _default_vp)
    vs_func = vs_func or _resolve_callable(config, "vs_m_s", _default_vs)
    density_func = density_func or _resolve_callable(config, "density_kg_m3", _default_density)

    # Evaluate at every GLL node
    vp = np.array([vp_func(x, y, z) for x, y, z in flat], dtype=np.float64).reshape(shape)
    vs = np.array([vs_func(x, y, z) for x, y, z in flat], dtype=np.float64).reshape(shape)
    density = np.array([density_func(x, y, z) for x, y, z in flat], dtype=np.float64).reshape(
        shape
    )

    return vp, vs, density


def _resolve_callable(
    config: types.ModuleType | None,
    attr_name: str,
    fallback: Callable[[float, float, float], float],
) -> Callable[[float, float, float], float]:
    """Return config attribute if it is callable, otherwise fallback."""
    if config is not None:
        attr = getattr(config, attr_name, None)
        if callable(attr):
            return attr
    return fallback


def _default_vp(_x: float, _y: float, _z: float) -> float:
    return 3000.0


def _default_vs(_x: float, _y: float, _z: float) -> float:
    return 1500.0


def _default_density(_x: float, _y: float, _z: float) -> float:
    return 2500.0
