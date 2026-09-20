"""Regression tests for the SEM artifact consistency checker."""

import importlib.util
import sys
from pathlib import Path

import h5py


SCRIPT_PATH = Path(__file__).parents[1] / "examples" / "_shared" / "check_sem_consistency.py"
SPEC = importlib.util.spec_from_file_location("check_sem_consistency", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECK_SEM_CONSISTENCY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_SEM_CONSISTENCY)


def test_current_cell_schema_and_duration_step_count_pass(tmp_path, monkeypatch):
    (tmp_path / "config.py").write_text(
        """\
lx = ly = lz = 10.0
source_x_m = source_y_m = source_z_m = 5.0
output_dt_s = 0.1
total_duration_s = 0.2
source_force_amplitude_n = 1.0
def stf_func(t_s):
    return 1.0
def vp_m_s(x_m, y_m, z_m):
    return 5000.0
def vs_m_s(x_m, y_m, z_m):
    return 3000.0
def density_kg_m3(x_m, y_m, z_m):
    return 2700.0
"""
    )
    with h5py.File(tmp_path / "config.h5", "w") as config:
        source = config.create_group("source")
        source.attrs.update({"x": 5.0, "y": 5.0, "z": 5.0})
        source.create_dataset("stf_t", data=[0.0, 0.1], compression=None)
        source.create_dataset("stf_values", data=[1.0, 1.0], compression=None)
        simulation = config.create_group("simulation")
        simulation.attrs.update({"output_dt_s": 0.1, "solver_dt": 0.1, "snapshot_stride": 1})
    with h5py.File(tmp_path / "model.h5", "w") as model:
        cell = model.create_group("field/cell")
        cell.create_dataset("vp", data=[5000.0], compression=None)
        cell.create_dataset("vs", data=[3000.0], compression=None)
        cell.create_dataset("density", data=[2700.0], compression=None)

    monkeypatch.setattr(sys, "argv", ["check_sem_consistency.py", str(tmp_path)])

    assert CHECK_SEM_CONSISTENCY.main() == 0
