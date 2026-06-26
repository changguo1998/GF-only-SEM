import os
import sys
import tempfile

import h5py
import numpy as np

_project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _project_root)

from preprocess.config_writer import write_config


def _make_mock_config():
    """Create a mock config module with new field names and unit suffixes."""
    from types import ModuleType

    mod = ModuleType("mock_config")
    mod.title = "test_simulation"
    mod.polynomial_order = 3
    mod.output_dt_s = 0.001
    mod.nsteps = 100
    mod.cfl_safety = 0.5
    mod.snapshot_precision = "float32"
    mod.storage_limit_gb = 100.0
    return mod


class TestConfigWriter:
    def test_writes_simulation_group(self):
        config = _make_mock_config()
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}
        stf_t = np.arange(100, dtype=np.float64) * 0.001
        stf_values = np.sin(stf_t * 2 * np.pi * 5, dtype=np.float64)

        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "configs", "config.h5")
            write_config(config_path, config, domain_bounds, stf_t, stf_values)

            with h5py.File(config_path, "r") as f:
                assert "simulation" in f
                sim = f["simulation"]
                assert sim.attrs["title"] == "test_simulation"
                assert sim.attrs["polynomial_order"] == 3
                # solver_dt falls back to config_module.output_dt_s
                assert sim.attrs["solver_dt"] == 0.001
                assert sim.attrs["output_dt_s"] == 0.001
                # snapshot_stride defaults to 1 when not passed
                assert sim.attrs["snapshot_stride"] == 1
                assert sim.attrs["nsteps"] == 100
                assert sim.attrs["cfl_safety"] == 0.5
                assert sim.attrs["snapshot_precision"] == "float32"
                assert sim.attrs["storage_limit_gb"] == 100.0

    def test_writes_domain_group(self):
        config = _make_mock_config()
        domain_bounds = {
            "xmin": -1.0,
            "xmax": 2.0,
            "ymin": -0.5,
            "ymax": 1.5,
            "zmin": 0.0,
            "zmax": 3.0,
        }
        stf_t = np.arange(100, dtype=np.float64) * 0.001
        stf_values = np.sin(stf_t)

        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "configs", "config.h5")
            write_config(config_path, config, domain_bounds, stf_t, stf_values)

            with h5py.File(config_path, "r") as f:
                assert "domain" in f
                dom = f["domain"]
                for key, val in domain_bounds.items():
                    assert dom.attrs[key] == val, f"{key}: {dom.attrs[key]} != {val}"

    def test_writes_source_group(self):
        config = _make_mock_config()
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}
        stf_t = np.arange(100, dtype=np.float64) * 0.001
        stf_values = np.sin(stf_t * 2 * np.pi * 5, dtype=np.float64)
        source_xyz = np.array([500.0, 500.0, 0.0], dtype=np.float64)

        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "configs", "config.h5")
            write_config(config_path, config, domain_bounds, stf_t, stf_values, source_xyz)

            with h5py.File(config_path, "r") as f:
                assert "source" in f
                src = f["source"]
                assert np.array_equal(src["stf_t"][:], stf_t)
                assert np.array_equal(src["stf_values"][:], stf_values)
                assert src.attrs["x"] == 500.0
                assert src.attrs["y"] == 500.0
                assert src.attrs["z"] == 0.0

    def test_source_without_xyz(self):
        config = _make_mock_config()
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}
        stf_t = np.arange(100, dtype=np.float64) * 0.001
        stf_values = np.sin(stf_t)

        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "configs", "config.h5")
            write_config(config_path, config, domain_bounds, stf_t, stf_values, source_xyz=None)

            with h5py.File(config_path, "r") as f:
                assert "source" in f
                assert "x" not in f["source"].attrs

    def test_create_parent_dirs(self):
        config = _make_mock_config()
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}
        stf_t = np.array([0.0, 0.001], dtype=np.float64)
        stf_values = np.array([0.0, 0.1], dtype=np.float64)

        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "deeply", "nested", "config.h5")
            write_config(config_path, config, domain_bounds, stf_t, stf_values)
            assert os.path.isfile(config_path)

    def test_stf_values_are_float64(self):
        config = _make_mock_config()
        domain_bounds = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 2}
        stf_t = np.arange(100, dtype=np.float64)
        stf_values = np.arange(100, dtype=np.float64)

        with tempfile.TemporaryDirectory() as td:
            config_path = os.path.join(td, "configs", "config.h5")
            write_config(config_path, config, domain_bounds, stf_t, stf_values)

            with h5py.File(config_path, "r") as f:
                assert f["source"]["stf_t"].dtype == np.float64
                assert f["source"]["stf_values"].dtype == np.float64
