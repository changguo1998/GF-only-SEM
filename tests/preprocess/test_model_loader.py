"""Tests for model_loader module."""

import os
import sys
import types

import numpy as np
import pytest

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.model_loader import load_and_interpolate


class TestModelLoader:
    def test_returns_placeholder_values(self):
        """Placeholder model_loader returns default vp=3000, vs=1500, density=2500."""
        gll_coords = np.zeros((2, 4, 4, 4, 3), dtype=np.float64)
        vp, vs, density = load_and_interpolate(None, gll_coords)
        assert vp.shape == (2, 4, 4, 4)
        assert vs.shape == (2, 4, 4, 4)
        assert density.shape == (2, 4, 4, 4)
        assert np.allclose(vp, 3000.0)
        assert np.allclose(vs, 1500.0)
        assert np.allclose(density, 2500.0)

    def test_float64_type(self):
        gll_coords = np.ones((1, 3, 3, 3, 3), dtype=np.float64)
        vp, vs, density = load_and_interpolate("dummy.h5", gll_coords)
        assert vp.dtype == np.float64
        assert vs.dtype == np.float64
        assert density.dtype == np.float64

    def test_material_arrays_full_shape(self):
        """config.material_arrays with full-shape arrays bypasses callable evaluation."""
        shape = (2, 3, 3, 3)  # n_cell=2, NGLL=3
        gll_coords = np.zeros((*shape, 3), dtype=np.float64)
        cfg = types.SimpleNamespace(
            material_arrays={
                "vp": np.full(shape, 4200.0),
                "vs": np.full(shape, 2200.0),
                "density": np.full(shape, 2900.0),
            }
        )
        vp, vs, density = load_and_interpolate(None, gll_coords, config=cfg)  # type: ignore[arg-type]
        assert vp.shape == shape
        assert np.allclose(vp, 4200.0)
        assert np.allclose(vs, 2200.0)
        assert np.allclose(density, 2900.0)

    def test_material_arrays_scalar_broadcast(self):
        """Scalar values in material_arrays are broadcast to GLL field shape."""
        shape = (1, 4, 4, 4)
        gll_coords = np.zeros((*shape, 3), dtype=np.float64)
        cfg = types.SimpleNamespace(
            material_arrays={"vp": 5000.0, "vs": 3000.0, "density": 2700.0}
        )
        vp, vs, density = load_and_interpolate(None, gll_coords, config=cfg)  # type: ignore[arg-type]
        assert np.allclose(vp, 5000.0)
        assert np.allclose(vs, 3000.0)
        assert np.allclose(density, 2700.0)
        assert vp.shape == shape

    def test_material_arrays_shape_mismatch_raises(self):
        """Shape mismatch in material_arrays raises ValueError."""
        shape = (2, 5, 5, 5)
        gll_coords = np.zeros((*shape, 3), dtype=np.float64)
        cfg = types.SimpleNamespace(
            material_arrays={
                "vp": np.zeros((3, 5, 5, 5)),  # wrong n_cell
                "vs": 3000.0,
                "density": 2700.0,
            }
        )
        with pytest.raises(ValueError, match="shape.*does not match"):
            load_and_interpolate(None, gll_coords, config=cfg)  # type: ignore[arg-type]

    def test_material_arrays_partial_fallback(self):
        """Partial material_arrays: missing keys fall back to config callable or default."""
        shape = (1, 3, 3, 3)
        gll_coords = np.zeros((*shape, 3), dtype=np.float64)
        cfg = types.SimpleNamespace(
            material_arrays={"vp": 4200.0, "vs": 2200.0}  # no density → default
        )
        vp, vs, density = load_and_interpolate(None, gll_coords, config=cfg)  # type: ignore[arg-type]
        assert np.allclose(vp, 4200.0)
        assert np.allclose(vs, 2200.0)
        assert np.allclose(density, 2500.0)  # default

    def test_material_arrays_overridden_by_explicit_callable(self):
        """Explicit vp_func/vs_func/density_func override material_arrays."""
        shape = (1, 3, 3, 3)
        gll_coords = np.zeros((*shape, 3), dtype=np.float64)
        cfg = types.SimpleNamespace(
            material_arrays={"vp": 1000.0, "vs": 1000.0, "density": 1000.0}
        )

        def vp_override(_x, _y, _z):
            return 9999.0  # type: ignore[return-value]

        vp, vs, density = load_and_interpolate(
            None,
            gll_coords,
            config=cfg,  # type: ignore[arg-type]
            vp_func=vp_override,
        )
        # Explicit vp_func wins; vs and density still come from material_arrays
        assert np.allclose(vp, 9999.0)
        assert np.allclose(vs, 1000.0)
        assert np.allclose(density, 1000.0)
