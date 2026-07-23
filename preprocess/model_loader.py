"""Model loader — evaluates config material functions or arrays at GLL nodes.

Reads Vp, Vs, and density from config.py. Three modes (checked in order):

1. ``config.material_arrays`` dict — direct numpy arrays, skip evaluation.
   Keys: ``"vp"``, ``"vs"``, ``"density"``. Each value may be a scalar
   (broadcast to full field) or an array matching the GLL field shape
   ``(n_cell, NGLL, NGLL, NGLL)``.

2. ``config.vp_m_s`` / ``config.vs_m_s`` / ``config.density_kg_m3``
   callables — evaluated at every GLL node (vectorized then scalar fallback).

3. Hardcoded defaults (Vp=3000 m/s, Vs=1500 m/s, density=2500 kg/m³).
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
    """Evaluate material model at GLL node positions.

    Priority: explicit ``vp_func`` / ``vs_func`` / ``density_func`` args >
    ``config.material_arrays`` dict > ``config.vp_m_s`` / ``config.vs_m_s`` /
    ``config.density_kg_m3`` callables > hardcoded defaults.

    Args:
        model_path: Ignored — material is defined purely in config.
        gll_coords: GLL node coordinates, shape [n_cell, NGLL, NGLL, NGLL, 3].
        config: Config module (e.g. from load_config / config.py import).
        vp_func: Explicit P-wave velocity callable.  Takes precedence over
            ``config.vp_m_s`` and ``config.material_arrays``.
        vs_func: Explicit S-wave velocity callable.
        density_func: Explicit density callable.

    Returns:
        Tuple of (vp, vs, density), each [n_cell, NGLL, NGLL, NGLL] float64.
    """
    shape = gll_coords.shape[:-1]  # (n_cell, NGLL, NGLL, NGLL)
    material_arrays = getattr(config, "material_arrays", None) if config is not None else None

    vp = _resolve_one_field("vp", shape, gll_coords, vp_func, material_arrays, config)
    vs = _resolve_one_field("vs", shape, gll_coords, vs_func, material_arrays, config)
    density = _resolve_one_field(
        "density", shape, gll_coords, density_func, material_arrays, config
    )
    return vp, vs, density


def _resolve_one_field(
    key: str,
    shape: tuple[int, ...],
    gll_coords: npt.NDArray[np.float64],
    explicit_func: Callable | None,
    material_arrays: dict | None,
    config: types.ModuleType | None,
) -> npt.NDArray[np.float64]:
    """Resolve a single material field with cascading priority.

    Priority (per-field): explicit callable > material_arrays[key] >
    config callable > hardcoded default.
    """
    flat = gll_coords.reshape(-1, 3)

    # 1) Explicit per-field callable
    if explicit_func is not None:
        return _evaluate_field(flat, explicit_func, shape)

    # 2) Direct array from material_arrays
    if material_arrays is not None and key in material_arrays:
        return _resolve_array_field(material_arrays, key, shape)

    # 3) Config callable attribute
    _attr_map = {"vp": "vp_m_s", "vs": "vs_m_s", "density": "density_kg_m3"}
    _default_map = {"vp": _default_vp, "vs": _default_vs, "density": _default_density}
    callable_func = _resolve_callable(config, _attr_map[key], _default_map[key])
    return _evaluate_field(flat, callable_func, shape)


def _evaluate_field(
    flat_coords: npt.NDArray[np.float64], func: Callable, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Evaluate a material callable at all GLL nodes.

    Tries array-compatible call first (func(x_arr, y_arr, z_arr) returning
    array or broadcastable scalar). Falls back to scalar iteration if the
    callable raises TypeError (e.g., uses ``if`` instead of ``np.where``).
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


def _resolve_array_field(
    material_arrays: dict, key: str, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Extract a material field from ``material_arrays`` dict and validate shape.

    Supports:
    - Full array matching ``shape`` → returned as float64
    - Scalar value → broadcast to ``shape``
    - Missing key → raises ValueError with available keys
    """
    if key not in material_arrays:
        available = ", ".join(sorted(material_arrays.keys()))
        raise ValueError(f"material_arrays missing key '{key}'. Available keys: {available}")
    arr = np.asarray(material_arrays[key], dtype=np.float64)
    if arr.ndim == 0:
        return np.full(shape, arr.item(), dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(
            f"material_arrays['{key}'] shape {arr.shape} does not match "
            f"GLL field shape {shape} (n_cell, NGLL, NGLL, NGLL)"
        )
    return arr


def _resolve_callable(
    config: types.ModuleType | None,
    attr_name: str,
    fallback: Callable[[float, float, float], float],
) -> Callable[[float, float, float], float]:
    """Return config attribute if it is callable, otherwise fallback."""
    if config is not None:
        attr = getattr(config, attr_name, None)
        if callable(attr):
            return cast(Callable[[float, float, float], float], attr)
    return fallback


def _default_vp(_x: float, _y: float, _z: float) -> float:
    return 3000.0


def _default_vs(_x: float, _y: float, _z: float) -> float:
    return 1500.0


def _default_density(_x: float, _y: float, _z: float) -> float:
    return 2500.0
