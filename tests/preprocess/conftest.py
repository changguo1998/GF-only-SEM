"""Shared fixtures for preprocess tests."""

from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def config_dict():
    """Returns a dict that can be used to build a mock config module."""
    return {
        "title": "test_run",
        "polynomial_order": 3,
        "output_dt_s": 0.001,
        "total_duration_s": 0.5,
        "cfl_safety": 0.5,
        "snapshot_precision": "float32",
        "storage_limit_gb": 100,
        "n_ranks": 4,
        "pml_thickness": {"xmin": 3, "xmax": 3, "ymin": 3, "ymax": 3, "zmin": 0, "zmax": 3},
        "source_x_m": 500.0,
        "source_y_m": 500.0,
    }


@pytest.fixture
def mock_config_module(config_dict):
    """Creates a mock Python module with config attributes + callables."""
    mod = ModuleType("mock_config")
    for key, val in config_dict.items():
        setattr(mod, key, val)

    def stf_func(t):
        import numpy as np

        return (1 - 2 * (np.pi * 5.0 * (t - 0.3)) ** 2) * np.exp(-((np.pi * 5.0 * (t - 0.3)) ** 2))

    def vp_m_s(x, y, z):
        return 3000.0

    def vs_m_s(x, y, z):
        return 1500.0

    def density_kg_m3(x, y, z):
        return 2500.0

    mod.stf_func = stf_func
    mod.vp_m_s = vp_m_s
    mod.vs_m_s = vs_m_s
    mod.density_kg_m3 = density_kg_m3
    return mod


@pytest.fixture
def tmp_dir():
    """Temporary directory that auto-cleans after test."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
