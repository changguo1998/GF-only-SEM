"""Model loader — evaluates config material callables at GLL nodes.

The only user-facing API is three callables in config.py::

    def vp_m_s(x_m, y_m, z_m) -> float | np.ndarray:
        ...
    def vs_m_s(x_m, y_m, z_m) -> float | np.ndarray:
        ...
    def density_kg_m3(x_m, y_m, z_m) -> float | np.ndarray:
        ...

Users are free to implement these callables however they want —
formulas, h5py reads, scipy interpolation, pandas CSV, etc.
The project imposes no data-format constraints.

If no config module is provided, hardcoded defaults are used
(Vp=3000 m/s, Vs=1500 m/s, density=2500 kg/m³).
"""

import types
from collections.abc import Callable
from typing import cast

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
    """Evaluate material model at every GLL node.

    Priority (per-field): explicit ``*_func`` argument > config.py callable >
    hardcoded default.  Each of vp, vs, density is resolved independently.

    Args:
        model_path: Ignored — material is defined purely in config.
        gll_coords: GLL node coordinates, shape ``[n_cell, NGLL, NGLL, NGLL, 3]``.
        config: Config module imported from ``config.py``.  Its ``vp_m_s``,
            ``vs_m_s``, and ``density_kg_m3`` attributes are used if they
            exist and are callable.
        vp_func: Explicit override for P-wave velocity.
        vs_func: Explicit override for S-wave velocity.
        density_func: Explicit override for density.

    Returns:
        ``(vp, vs, density)`` — each ``[n_cell, NGLL, NGLL, NGLL]`` float64.
    """
    shape = gll_coords.shape[:-1]

    vp = _resolve_one_field("vp", shape, gll_coords, vp_func, config)
    vs = _resolve_one_field("vs", shape, gll_coords, vs_func, config)
    density = _resolve_one_field("density", shape, gll_coords, density_func, config)
    return vp, vs, density


def _resolve_one_field(
    key: str,
    shape: tuple[int, ...],
    gll_coords: npt.NDArray[np.float64],
    explicit_func: Callable | None,
    config: types.ModuleType | None,
) -> npt.NDArray[np.float64]:
    """Resolve a single material field: explicit > config callable > default."""
    flat = gll_coords.reshape(-1, 3)

    if explicit_func is not None:
        return _evaluate_field(flat, explicit_func, shape)

    _attr_map = {"vp": "vp_m_s", "vs": "vs_m_s", "density": "density_kg_m3"}
    _default_map = {"vp": _default_vp, "vs": _default_vs, "density": _default_density}

    callable_func = _resolve_callable(config, _attr_map[key], _default_map[key])
    return _evaluate_field(flat, callable_func, shape)


def _evaluate_field(
    flat_coords: npt.NDArray[np.float64], func: Callable, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Evaluate a material callable at all GLL nodes.

    Tries vectorized call first (``func(x_arr, y_arr, z_arr)`` returning an
    array or broadcastable scalar), then falls back to per-point scalar
    iteration when the callable uses ``if`` / ``else`` control flow.
    """
    try:
        result = func(flat_coords[:, 0], flat_coords[:, 1], flat_coords[:, 2])
        arr = np.asarray(result, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(shape, arr.item())
        return arr.reshape(shape)
    except (TypeError, ValueError):
        return np.array([func(x, y, z) for x, y, z in flat_coords], dtype=np.float64).reshape(
            shape
        )


def _resolve_callable(
    config: types.ModuleType | None,
    attr_name: str,
    fallback: Callable[[float, float, float], float],
) -> Callable[[float, float, float], float]:
    """Return ``config.<attr_name>`` if callable, otherwise *fallback*."""
    if config is not None:
        attr = getattr(config, attr_name, None)
        if callable(attr):
            return cast(Callable[[float, float, float], float], attr)
    return fallback


# ── Hardcoded defaults (used when no config.py provided) ──────────────────


def _default_vp(_x: float, _y: float, _z: float) -> float:
    return 3000.0


def _default_vs(_x: float, _y: float, _z: float) -> float:
    return 1500.0


def _default_density(_x: float, _y: float, _z: float) -> float:
    return 2500.0
