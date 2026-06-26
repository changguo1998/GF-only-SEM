"""Tests for STF (Source Time Function) evaluator module."""

import os
import sys

import numpy as np

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.stf_evaluator import evaluate_stf


def mock_ricker_wavelet(t: float) -> float:
    """Ricker wavelet with f0=5 Hz, peak at t0=0.3 s."""
    f0 = 5.0
    t0 = 0.3
    return (1 - 2 * (np.pi * f0 * (t - t0)) ** 2) * np.exp(-((np.pi * f0 * (t - t0)) ** 2))


def mock_step_function(t: float) -> float:
    """Simple Heaviside step: 0 for t < 0, 1 for t >= 0."""
    return 1.0 if t >= 0.0 else 0.0


class TestEvaluateSTF:
    """Test evaluate_stf function."""

    def test_output_shape_matches_nsteps(self):
        """Returned arrays should have shape (nsteps,)."""
        dt = 0.001
        nsteps = 100
        stf_t, stf_vals = evaluate_stf(mock_ricker_wavelet, dt, nsteps)

        assert stf_t.shape == (nsteps,)
        assert stf_vals.shape == (nsteps,)
        assert stf_t.dtype == np.float64
        assert stf_vals.dtype == np.float64

    def test_times_start_at_zero(self):
        """Time values should start at t=0 and increment by dt."""
        dt = 0.001
        nsteps = 10
        stf_t, _ = evaluate_stf(mock_ricker_wavelet, dt, nsteps)

        for i in range(nsteps):
            assert np.isclose(stf_t[i], i * dt), f"t[{i}] = {stf_t[i]}, expected {i * dt}"

    def test_correct_ricker_values(self):
        """Ricker wavelet evaluated at known points should match expected values."""
        dt = 0.001
        nsteps = 1000  # enough to capture the peak at t0=0.3
        stf_t, stf_vals = evaluate_stf(mock_ricker_wavelet, dt, nsteps)

        # At t = 0.3 (peak of Ricker), value should be 1.0
        closest_idx = np.argmin(np.abs(stf_t - 0.3))
        assert np.isclose(stf_vals[int(closest_idx)], 1.0, atol=1e-6)

    def test_stf_values_at_t0_peak(self):
        """At t = t0, Ricker should evaluate to exactly 1.0."""
        dt = 0.0001
        t0 = 0.3
        nsteps = int(round(t0 / dt))
        stf_t, stf_vals = evaluate_stf(mock_ricker_wavelet, dt, nsteps + 1)

        assert np.isclose(stf_vals[nsteps], 1.0)

    def test_step_function_correct(self):
        """Step function should return 1.0 for all t >= 0."""
        dt = 0.001
        nsteps = 50
        _, stf_vals = evaluate_stf(mock_step_function, dt, nsteps)
        assert np.allclose(stf_vals, 1.0)

    def test_small_nsteps(self):
        """nsteps=1 should return single value at t=0."""
        dt = 0.001
        stf_t, stf_vals = evaluate_stf(mock_ricker_wavelet, dt, 1)
        assert stf_t.shape == (1,)
        assert np.isclose(stf_t[0], 0.0)
        # Ricker at t=0: (1 - 2*(pi*5*(-0.3))^2) * exp(-(pi*5*(-0.3))^2)
        expected = (1 - 2 * (np.pi * 5 * (-0.3)) ** 2) * np.exp(-((np.pi * 5 * (-0.3)) ** 2))
        assert np.isclose(stf_vals[0], expected)

    def test_large_nsteps(self):
        """Large nsteps should complete without error and produce correct time range."""
        dt = 0.001
        nsteps = 10000
        stf_t, stf_vals = evaluate_stf(mock_ricker_wavelet, dt, nsteps)

        assert stf_t[0] == 0.0
        assert np.isclose(stf_t[-1], (nsteps - 1) * dt)
        # Ricker decays: at t=10 s the value should be essentially 0
        assert abs(stf_vals[-1]) < 1e-10

    def test_different_dt_values(self):
        """Different dt should give different time spacing."""
        nsteps = 10
        _, vals1 = evaluate_stf(mock_ricker_wavelet, 0.001, nsteps)
        _, vals2 = evaluate_stf(mock_ricker_wavelet, 0.01, nsteps)

        # With dt=0.01, we skip the peak (at 0.3) and evaluate at coarser grid
        # Values should differ since we're sampling at different points
        assert not np.allclose(vals1, vals2, atol=1e-6)
