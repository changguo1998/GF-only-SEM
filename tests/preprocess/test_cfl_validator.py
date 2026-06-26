"""Tests for cfl_validator module."""

import os
import sys

import numpy as np
import pytest

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.cfl_validator import compute_cfl_dt, compute_solver_dt


class TestCFLValidator:
    """Tests for CFL validation and solver_dt derivation."""

    def test_compute_cfl_dt_basic(self):
        """Test compute_cfl_dt with a simple cube mesh."""
        n_cell, n_gll = 1, 4
        gll_coords = np.zeros((n_cell, n_gll, n_gll, n_gll, 3), dtype=np.float64)
        vp_array = np.full((n_cell, n_gll, n_gll, n_gll), 5000.0, dtype=np.float64)

        # Simple 1x1x1 cube from 0 to 1
        for i in range(n_gll):
            for j in range(n_gll):
                for k in range(n_gll):
                    gll_coords[0, i, j, k] = [i / 3.0, j / 3.0, k / 3.0]

        cfl_safety = 0.5
        cfl_dt = compute_cfl_dt(gll_coords, vp_array, cfl_safety)
        expected_cfl_dt = 0.5 * (1.0 / 3.0) / 5000.0  # h_min ≈ 1/3, vp_max = 5000
        assert abs(cfl_dt - expected_cfl_dt) < 1e-12, f"{cfl_dt} != {expected_cfl_dt}"

    def test_compute_cfl_dt_negative_vp_raises(self):
        """Test that negative vp raises ValueError."""
        n_cell, n_gll = 1, 4
        gll_coords = np.zeros((n_cell, n_gll, n_gll, n_gll, 3), dtype=np.float64)
        for i in range(n_gll):
            for j in range(n_gll):
                for k in range(n_gll):
                    gll_coords[0, i, j, k] = [i / 3.0, j / 3.0, k / 3.0]
        vp_array = np.full((n_cell, n_gll, n_gll, n_gll), -1.0, dtype=np.float64)

        with pytest.raises(ValueError, match="Invalid maximum vp"):
            compute_cfl_dt(gll_coords, vp_array, 0.5)

    def test_compute_solver_dt_exact_stride(self):
        """Test compute_solver_dt finds exact stride dividing output_dt_s."""
        output_dt_s = 0.01
        cfl_dt = 0.00173
        solver_dt, stride = compute_solver_dt(output_dt_s, cfl_dt)
        assert solver_dt == pytest.approx(output_dt_s / stride)
        assert stride == 6  # 0.01/6 ≈ 0.001667 ≤ 0.00173

    def test_compute_solver_dt_stride_1(self):
        """Test when output_dt_s ≤ cfl_dt, stride should be 1."""
        output_dt_s = 0.001
        cfl_dt = 0.01
        solver_dt, stride = compute_solver_dt(output_dt_s, cfl_dt)
        assert stride == 1
        assert solver_dt == output_dt_s

    def test_compute_solver_dt_no_valid_stride(self):
        """Test when no stride works within MAX_STRIDE."""
        output_dt_s = 100.0
        cfl_dt = 0.001
        with pytest.raises(ValueError, match="output_dt_s"):
            compute_solver_dt(output_dt_s, cfl_dt)

    def test_compute_solver_dt_custom_max_stride(self):
        """Test custom max_stride parameter."""
        output_dt_s = 0.1
        cfl_dt = 0.0005
        # With max_stride=200, output_dt_s/cfl_dt demands stride ~200 for 0.1/0.0005
        solver_dt, stride = compute_solver_dt(output_dt_s, cfl_dt, max_stride=250)
        assert stride <= 250
        assert stride == 200
        assert solver_dt == pytest.approx(output_dt_s / stride)

    def test_compute_solver_dt_exact_equality(self):
        """Test when output_dt_s / stride == cfl_dt exactly."""
        output_dt_s = 0.005
        cfl_dt = 0.001
        solver_dt, stride = compute_solver_dt(output_dt_s, cfl_dt)
        assert stride == 5
        assert solver_dt == pytest.approx(cfl_dt)

    def test_compute_solver_dt_large_cfl_dt(self):
        """Test with very large cfl_dt (should give stride=1)."""
        output_dt_s = 0.1
        cfl_dt = 10.0
        solver_dt, stride = compute_solver_dt(output_dt_s, cfl_dt)
        assert stride == 1
        assert solver_dt == output_dt_s

    def test_compute_solver_dt_negative_output_dt_raises(self):
        """Test negative output_dt_s raises ValueError."""
        with pytest.raises(ValueError, match="output_dt_s"):
            compute_solver_dt(-0.01, 0.001)

    def test_compute_solver_dt_zero_output_dt_raises(self):
        """Test zero output_dt_s raises ValueError."""
        with pytest.raises(ValueError, match="output_dt_s"):
            compute_solver_dt(0.0, 0.001)

    def test_compute_solver_dt_negative_cfl_dt_raises(self):
        """Test negative cfl_dt raises ValueError."""
        with pytest.raises(ValueError, match="cfl_dt"):
            compute_solver_dt(0.01, -1.0)

    def test_compute_solver_dt_invalid_max_stride_raises(self):
        """Test invalid max_stride raises ValueError."""
        with pytest.raises(ValueError, match="max_stride"):
            compute_solver_dt(0.01, 0.001, max_stride=0)
