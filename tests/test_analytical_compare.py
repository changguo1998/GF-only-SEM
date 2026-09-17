"""Regression tests for the full-space analytical comparison."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "_shared" / "analytical_compare.py"
SPEC = importlib.util.spec_from_file_location("analytical_compare", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYTICAL_COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYTICAL_COMPARE)


def test_amplitude_scale_rejects_threefold_postprocess_error():
    assert ANALYTICAL_COMPARE.amplitude_scale_is_acceptable(1.0)
    assert not ANALYTICAL_COMPARE.amplitude_scale_is_acceptable(1.0 / 3.0)


def test_best_fit_scale_uses_sem_over_analytical_convention():
    analytical = np.array([1.0, -2.0, 3.0])
    sem = 1.1 * analytical
    assert np.isclose(ANALYTICAL_COMPARE.best_fit_scale(sem, analytical), 1.1)


def test_scale_fitted_l2_is_invariant_to_sem_amplitude():
    analytical = np.array([1.0, -2.0, 3.0, -1.0])
    sem = np.array([1.0, -1.8, 3.2, -0.9])
    _, original_error = ANALYTICAL_COMPARE.scale_fitted_relative_l2(sem, analytical)
    _, scaled_error = ANALYTICAL_COMPARE.scale_fitted_relative_l2(3.0 * sem, analytical)
    assert np.isclose(original_error, scaled_error)


def test_fixed_receivers_does_not_skip_following_option(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["analytical_compare.py", "greenfun", "--fixed-receivers", "points.npy", "--fullspace"],
    )
    args = ANALYTICAL_COMPARE.parse_args()
    assert args["fixed_receivers"] == "points.npy"
    assert args["fullspace"] is True
