"""Regression checks for fine reference sampling and comparison acceptance."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.special import erf

from examples.halfspace import reference
from examples.halfspace.reference_sampling import (
    converged_displacement,
    integrate_batched,
    sampled_displacement,
)


@pytest.mark.parametrize("depth_m", [1.0, 10.0, 500.0])
@pytest.mark.parametrize("azimuth", [0.0, 0.4])
def test_batched_step_matches_original(depth_m, azimuth):
    receiver = [14000 * np.cos(azimuth), 14000 * np.sin(azimuth), 0.0]
    arguments = (5000.0, 3000.0, 2700.0, [0.0, 0.0, depth_m], receiver, [3.25, 4.75, 5.1])
    scalar = reference.compute_step_response(*arguments, n_quad_pts=12, n_seg=10)
    batched = reference.compute_step_response(
        *arguments, n_quad_pts=12, n_seg=10, integrate_segments=integrate_batched
    )
    np.testing.assert_allclose(batched, scalar, rtol=1e-10, atol=1e-30)


def test_fft_sampling_matches_original_direct_convolution():
    time_s = np.arange(101) * 0.05
    force = np.exp(-(((time_s - 1.0) / 0.2) ** 2))
    physical = (5000.0, 3000.0, 2700.0, [0.0, 0.0, 100.0], [5000.0, 0.0, 0.0])
    step = reference.compute_step_response(*physical, time_s, n_quad_pts=8, n_seg=4)
    expected = reference.convolve_with_stf(step, force, 0.05)
    actual = sampled_displacement(*physical, time_s, time_s, force, 1, n_quad_pts=8, n_seg=4)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-11


def test_fine_impulse_sampling_recovers_known_convolved_waveform(monkeypatch):
    arrival_s, impulse_width_s, source_width_s, source_peak_s = 0.1234, 0.002, 0.05, 0.4
    grids = []

    def narrow_step(*arguments, **kwargs):
        grid = arguments[5]
        grids.append(grid)
        step = np.zeros((len(grid), 3, 3))
        step[:, 2, 2] = 0.5 * (1 + erf((grid - arrival_s) / (np.sqrt(2) * impulse_width_s)))
        return step

    monkeypatch.setattr(reference, "compute_step_response", narrow_step)
    output_time = np.arange(21) * 0.05
    source_time = np.linspace(0, 1, 10001)
    source_values = np.exp(-0.5 * ((source_time - source_peak_s) / source_width_s) ** 2)
    arguments = (
        5000,
        3000,
        2700,
        [0, 0, 1],
        [14000, 0, 0],
        output_time,
        source_time,
        source_values,
    )
    coarse = sampled_displacement(*arguments, 1)[:, 2, 2]
    fine = sampled_displacement(*arguments, 32)[:, 2, 2]
    combined_width = np.hypot(source_width_s, impulse_width_s)
    expected = (
        source_width_s
        / combined_width
        * np.exp(-0.5 * ((output_time - source_peak_s - arrival_s) / combined_width) ** 2)
    )
    assert np.linalg.norm(coarse - expected) / np.linalg.norm(expected) > 0.05
    assert np.linalg.norm(fine - expected) / np.linalg.norm(expected) < 0.001
    assert len(grids[-1]) == (len(output_time) - 1) * 32 + 1
    np.testing.assert_allclose(grids[-1][::32], output_time)


@pytest.mark.parametrize(
    "times,refinement", [([0, 0.05, 0.11], 2), ([0.05, 0.1, 0.15], 2), ([0, 0.05, 0.1], 0)]
)
def test_reject_invalid_sampling(times, refinement):
    with pytest.raises(ValueError):
        sampled_displacement(
            5000, 3000, 2700, [0, 0, 1], [1, 0, 0], times, [0, 1], [0, 1], refinement
        )


def test_reference_convergence_uses_successive_reference_levels():
    levels = []

    def evaluate(refinement):
        levels.append(refinement)
        return np.ones((3, 3, 3)) * (1 + 1 / refinement**2)

    _, refinement, change = converged_displacement(evaluate, [0, 0.05, 0.1], 0.11, 1, 16, 0.02)
    assert levels == [1, 2, 4, 8, 16]
    assert refinement == 16 and change < 0.02
    with pytest.raises(RuntimeError, match="did not converge"):
        converged_displacement(evaluate, [0, 0.05, 0.1], 0.11, 1, 4, 0.002)


def test_cli_rejects_failed_far_point_even_when_11km_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    config = SimpleNamespace(
        source_x_m=0.0,
        source_y_m=0.0,
        source_z_m=1.0,
        validation_offsets_m=[11000.0, 14000.0],
        validation_end_times_s=[0.2, 0.2],
        validation_depths_m=[0.0],
    )
    monkeypatch.setitem(sys.modules, "config", config)
    path = Path(__file__).resolve().parents[1] / "examples/halfspace-50x50x25/compare_grid.py"
    spec = importlib.util.spec_from_file_location("depth_compare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CASE_DIR", tmp_path)
    monkeypatch.setattr(module, "_earliest_pml_return_s", lambda point: 10.0)
    time_s = np.arange(4) * 0.05
    analytical = np.arange(36.0).reshape(4, 3, 3)
    library = SimpleNamespace(
        query=lambda receiver, source, quantity: SimpleNamespace(
            time=time_s, displacement=analytical if receiver[0] == 11000 else -analytical
        )
    )
    monkeypatch.setattr(module, "GreenFunctionLibrary", lambda *args, **kwargs: library)
    monkeypatch.setattr(
        module, "converged_displacement", lambda *args, **kwargs: (analytical, 64, 0.0005)
    )
    monkeypatch.setattr(sys, "argv", [str(path)])
    with h5py.File(tmp_path / "config.h5", "w") as handle:
        source = handle.create_group("source")
        source.attrs.update(x=0.0, y=0.0, z=1.0)
        source.create_dataset("stf_t", data=time_s, compression=None)
        source.create_dataset("stf_values", data=np.ones(4), compression=None)
    assert module.main() == 1
    assert "reference_relative_change" in (tmp_path / "log/lamb_distance_depth.csv").read_text()
