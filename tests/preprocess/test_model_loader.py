"""Tests for model_loader module."""

import os
import sys
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