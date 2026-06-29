"""Tests for config_loader module."""

import os
import sys

import pytest

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.config_loader import ConfigValidationError, load_config


class TestLoadConfig:
    """Test loading and validating config modules."""

    def test_loads_module_and_verifies_attributes(self, tmp_dir, config_dict):
        """Loaded config module should have all expected attributes."""
        # Write a config.py to tmp_dir
        config_path = tmp_dir / "test_config.py"
        lines = ["import numpy as np\n"]
        for key, val in config_dict.items():
            if isinstance(val, str):
                lines.append(f'{key} = "{val}"\n')
            elif isinstance(val, dict):
                lines.append(f"{key} = {val!r}\n")
            else:
                lines.append(f"{key} = {val}\n")
        lines.append("def stf_func(t):\n")
        lines.append(
            "    return (1 - 2 * (np.pi * 5.0 * (t - 0.3))**2) * np.exp(-(np.pi * 5.0 * (t - 0.3))**2)\n"
        )
        lines.append("def vp_m_s(x, y, z): return 3000.0\n")
        lines.append("def vs_m_s(x, y, z): return 1500.0\n")
        lines.append("def density_kg_m3(x, y, z): return 2500.0\n")
        config_path.write_text("".join(lines))

        mod = load_config(str(config_path), validate=False)
        assert mod.title == "test_run"
        assert mod.polynomial_order == 3
        assert mod.stf_func(0.3) == pytest.approx(1.0)
        assert mod.vp_m_s(0, 0, 0) == 3000.0

    def test_missing_required_field_raises(self, tmp_dir):
        """Missing required field should raise ConfigValidationError."""
        config_path = tmp_dir / "missing_title.py"
        config_path.write_text("polynomial_order = 3\n")
        with pytest.raises(ConfigValidationError, match="title"):
            load_config(str(config_path))

    def test_invalid_polynomial_order_raises(self, tmp_dir):
        """Invalid polynomial_order should raise ConfigValidationError."""
        config_path = tmp_dir / "bad_n.py"
        config_path.write_text("""title = "test"
polynomial_order = 0
output_dt_s = 0.001
total_duration_s = 0.5
cfl_safety = 0.5
snapshot_precision = "float32"
storage_limit_gb = 100
n_ranks = 4
pml_thickness = {"xmin": 3, "xmax": 3, "ymin": 3, "ymax": 3, "zmin": 0, "zmax": 3}
source_x_m = 500.0
source_y_m = 500.0
record_depth_max_m = 1000.0
nx_elements = 16
ny_elements = 16
nz_elements = 8
tilex_elements = [5, 5]
tiley_elements = [5, 5]
def stf_func(t): return 1.0
def vp_m_s(x, y, z): return 3000.0
def vs_m_s(x, y, z): return 1500.0
def density_kg_m3(x, y, z): return 2500.0
""")
        with pytest.raises(ConfigValidationError, match="polynomial_order"):
            load_config(str(config_path))

    def test_missing_stf_func_raises(self, tmp_dir):
        """Missing stf_func should raise ConfigValidationError."""
        config_path = tmp_dir / "no_stf.py"
        config_path.write_text("""title = "test"
polynomial_order = 3
output_dt_s = 0.001
total_duration_s = 0.5
cfl_safety = 0.5
snapshot_precision = "float32"
storage_limit_gb = 100
n_ranks = 4
pml_thickness = {"xmin": 3, "xmax": 3, "ymin": 3, "ymax": 3, "zmin": 0, "zmax": 3}
source_x_m = 500.0
source_y_m = 500.0
record_depth_max_m = 1000.0
nx_elements = 16
ny_elements = 16
nz_elements = 8
tilex_elements = [5, 5]
tiley_elements = [5, 5]
def vp_m_s(x, y, z): return 3000.0
def vs_m_s(x, y, z): return 1500.0
def density_kg_m3(x, y, z): return 2500.0
""")
        with pytest.raises(ConfigValidationError, match="stf_func"):
            load_config(str(config_path))

    def test_invalid_snapshot_precision_raises(self, tmp_dir):
        """Invalid snapshot_precision should raise ConfigValidationError."""
        config_path = tmp_dir / "bad_precision.py"
        config_path.write_text("""title = "test"
polynomial_order = 3
output_dt_s = 0.001
total_duration_s = 0.5
cfl_safety = 0.5
snapshot_precision = "int8"
storage_limit_gb = 100
n_ranks = 4
pml_thickness = {"xmin": 3, "xmax": 3, "ymin": 3, "ymax": 3, "zmin": 0, "zmax": 3}
source_x_m = 500.0
source_y_m = 500.0
record_depth_max_m = 1000.0
nx_elements = 16
ny_elements = 16
nz_elements = 8
tilex_elements = [5, 5]
tiley_elements = [5, 5]
def stf_func(t): return 1.0
def vp_m_s(x, y, z): return 3000.0
def vs_m_s(x, y, z): return 1500.0
def density_kg_m3(x, y, z): return 2500.0
""")
        with pytest.raises(ConfigValidationError, match="snapshot_precision"):
            load_config(str(config_path))

    def test_negative_output_dt_raises(self, tmp_dir):
        """Negative output_dt_s should raise ConfigValidationError."""
        config_path = tmp_dir / "neg_dt.py"
        config_path.write_text("""title = "test"
polynomial_order = 3
output_dt_s = -0.001
total_duration_s = 0.5
cfl_safety = 0.5
snapshot_precision = "float32"
storage_limit_gb = 100
n_ranks = 4
pml_thickness = {"xmin": 3, "xmax": 3, "ymin": 3, "ymax": 3, "zmin": 0, "zmax": 3}
source_x_m = 500.0
source_y_m = 500.0
record_depth_max_m = 1000.0
nx_elements = 16
ny_elements = 16
nz_elements = 8
tilex_elements = [5, 5]
tiley_elements = [5, 5]
def stf_func(t): return 1.0
def vp_m_s(x, y, z): return 3000.0
def vs_m_s(x, y, z): return 1500.0
def density_kg_m3(x, y, z): return 2500.0
""")
        with pytest.raises(ConfigValidationError, match="output_dt_s"):
            load_config(str(config_path))
