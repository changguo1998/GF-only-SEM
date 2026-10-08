"""Check the SEM depth comparison's amplitude and compatibility semantics."""

import csv
import importlib.util
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("depth,expected", [("-1", None), ("1", 1.0), ("1000", 1000.0)])
def test_depth_environment_matches_cpp_surface_convention(monkeypatch, depth, expected):
    monkeypatch.setenv("GF_SOURCE_DEPTH_M", depth)
    path = Path(__file__).resolve().parents[1] / "examples/halfspace-50x50x25/config.py"
    assert runpy.run_path(str(path))["source_z_m"] == expected


@pytest.mark.parametrize("mismatch", [False, True])
def test_depth_comparison_preserves_amplitude_and_checks_contract(tmp_path, monkeypatch, mismatch):
    case = Path(__file__).resolve().parents[1] / "examples/halfspace-50x50x25"
    monkeypatch.syspath_prepend(str(case))
    path = case / "compare_sem_depths.py"
    spec = importlib.util.spec_from_file_location("sem_depth_comparison_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CASE", tmp_path)
    monkeypatch.setattr(
        module,
        "config",
        SimpleNamespace(validation_offsets_m=[2000.0], validation_end_times_s=[0.2]),
    )
    time_s = np.arange(4) * 0.05
    displacement = np.arange(36.0).reshape(4, 3, 3)

    def read_config(directory):
        depth = 0 if directory.name == "depth_0000m" else 1
        contract = {"order": 5 if mismatch and depth else 4}
        return np.array([0.0, 0.0, depth]), contract, time_s, np.ones(4)

    def library(directory):
        factor = 1 if directory.parent.name == "depth_0000m" else 2
        return SimpleNamespace(
            query=lambda *args, **kwargs: SimpleNamespace(
                time=time_s, displacement=factor * displacement
            )
        )

    monkeypatch.setattr(module, "read_config", read_config)
    monkeypatch.setattr(module, "GreenFunctionLibrary", library)
    output = tmp_path / "comparison.csv"
    monkeypatch.setattr(sys, "argv", [str(path), "--depths", "1", "--output", str(output)])
    if mismatch:
        with pytest.raises(ValueError, match="Simulation/domain mismatch"):
            module.main()
        assert not output.exists()
    else:
        module.main()
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        assert [row["force_direction"] for row in rows] == ["all", "x", "y", "z"]
        for row in rows:
            assert float(row["raw_relative_l2"]) == pytest.approx(1.0)
            assert float(row["fitted_relative_l2"]) == pytest.approx(0.0)
            assert float(row["scale"]) == pytest.approx(2.0)
            assert float(row["correlation"]) == pytest.approx(1.0)
