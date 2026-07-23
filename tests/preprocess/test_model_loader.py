"""Tests for model_loader module."""

import os
import sys
import types

import numpy as np

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.model_loader import load_and_interpolate


class TestModelLoader:
    def test_returns_default_values(self):
        """No config → returns hardcoded defaults (Vp=3000, Vs=1500, ρ=2500)."""
        gll_coords = np.zeros((2, 4, 4, 4, 3), dtype=np.float64)
        vp, vs, density = load_and_interpolate(None, gll_coords)
        assert vp.shape == (2, 4, 4, 4)
        assert np.allclose(vp, 3000.0)
        assert np.allclose(vs, 1500.0)
        assert np.allclose(density, 2500.0)

    def test_float64_type(self):
        """All fields must be float64."""
        gll_coords = np.ones((1, 3, 3, 3, 3), dtype=np.float64)
        vp, vs, density = load_and_interpolate("dummy.h5", gll_coords)
        assert vp.dtype == np.float64
        assert vs.dtype == np.float64
        assert density.dtype == np.float64

    def test_config_callables_are_used(self):
        """Config callables produce custom values."""
        shape = (1, 3, 3, 3)
        gll_coords = np.zeros((*shape, 3), dtype=np.float64)
        cfg = types.SimpleNamespace(
            vp_m_s=lambda x, y, z: np.full_like(x, 5000.0),  # type: ignore[arg-type]
            vs_m_s=lambda x, y, z: np.full_like(x, 3000.0),  # type: ignore[arg-type]
            density_kg_m3=lambda x, y, z: np.full_like(x, 2700.0),  # type: ignore[arg-type]
        )
        vp, vs, density = load_and_interpolate(None, gll_coords, config=cfg)  # type: ignore[arg-type]
        assert np.allclose(vp, 5000.0)
        assert np.allclose(vs, 3000.0)
        assert np.allclose(density, 2700.0)

    def test_explicit_func_overrides_config(self):
        """Explicit vp_func wins over config.vp_m_s."""
        shape = (1, 3, 3, 3)
        gll_coords = np.zeros((*shape, 3), dtype=np.float64)
        cfg = types.SimpleNamespace(
            vp_m_s=lambda x, y, z: np.full_like(x, 1000.0),  # type: ignore[arg-type]
            vs_m_s=lambda x, y, z: np.full_like(x, 2000.0),  # type: ignore[arg-type]
            density_kg_m3=lambda x, y, z: np.full_like(x, 3000.0),  # type: ignore[arg-type]
        )

        def vp_override(_x, _y, _z):
            return 9999.0  # type: ignore[return-value]

        vp, vs, density = load_and_interpolate(
            None,
            gll_coords,
            config=cfg,
            vp_func=vp_override,  # type: ignore[arg-type]
        )
        assert np.allclose(vp, 9999.0)  # overridden
        assert np.allclose(vs, 2000.0)  # from config
        assert np.allclose(density, 3000.0)  # from config

    def test_partial_config_callables_fall_back(self):
        """Missing config callable falls back to default for that field."""
        shape = (1, 3, 3, 3)
        gll_coords = np.zeros((*shape, 3), dtype=np.float64)
        cfg = types.SimpleNamespace(
            vp_m_s=lambda x, y, z: np.full_like(x, 4200.0)  # type: ignore[arg-type]
            # vs_m_s and density_kg_m3 deliberately missing
        )
        vp, vs, density = load_and_interpolate(None, gll_coords, config=cfg)  # type: ignore[arg-type]
        assert np.allclose(vp, 4200.0)
        assert np.allclose(vs, 1500.0)  # default
        assert np.allclose(density, 2500.0)  # default
