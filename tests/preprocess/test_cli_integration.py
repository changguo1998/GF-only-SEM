"""Integration checks for config.py-driven preprocess orchestration."""

from __future__ import annotations

import textwrap

import h5py
import meshio
import numpy as np

import preprocess.cli as preprocess_cli
from tools.gmsh_to_hdf5 import extract_topology, write_topology


def _write_single_hex_model(path) -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = [("hexahedron", np.arange(8, dtype=np.int64).reshape(1, 8))]
    write_topology(path, extract_topology(meshio.Mesh(vertices, cells)))


def test_python_config_remains_authoritative_with_cpp_accelerator(tmp_path, monkeypatch):
    """The Python entry point must not invoke the compile-time C++ run config."""
    _write_single_hex_model(tmp_path / "model.h5")
    (tmp_path / "config.py").write_text(
        textwrap.dedent(
            """
            import numpy as np

            title = "surface_source_config_authority"
            polynomial_order = 2
            output_dt_s = 0.01
            total_duration_s = 0.03
            cfl_safety = 0.5
            snapshot_precision = "float32"
            storage_limit_gb = 1.0
            record_depth_max_m = 1.0
            n_ranks = 1
            nx_elements = 1
            ny_elements = 1
            nz_elements = 1
            pml_thickness = {
                "xmin": 0, "xmax": 0, "ymin": 0,
                "ymax": 0, "zmin": 0, "zmax": 0,
            }
            tilex_elements = [1]
            tiley_elements = [1]
            log_stride = 1
            source_x_m = 0.3
            source_y_m = 0.4
            source_z_m = None
            f0_for_pml_hz = 1.0

            def stf_func(time_s):
                return 7.0 + np.asarray(time_s)

            def vp_m_s(x_m, y_m, z_m):
                return np.full_like(z_m, 2.0, dtype=np.float64)

            def vs_m_s(x_m, y_m, z_m):
                return np.full_like(z_m, 1.0, dtype=np.float64)

            def density_kg_m3(x_m, y_m, z_m):
                return np.full_like(z_m, 1.0, dtype=np.float64)
            """
        )
    )

    original_run_binary = preprocess_cli._run_binary
    invoked_subcommands: list[str] = []

    def reject_unified_run(binary, args, timeout=600, desc=""):
        invoked_subcommands.append(args[0])
        assert args[0] != "run"
        return original_run_binary(binary, args, timeout=timeout, desc=desc)

    monkeypatch.setattr(preprocess_cli, "_run_binary", reject_unified_run)
    monkeypatch.chdir(tmp_path)
    preprocess_cli.main()

    with h5py.File(tmp_path / "config.h5", "r") as config_file:
        source = config_file["source"]
        weights = np.asarray(source["cells/weights"])
        stf_time = np.asarray(source["stf_t"])
        stf_values = np.asarray(source["stf_values"])

        assert source.attrs["x"] == 0.3
        assert source.attrs["y"] == 0.4
        assert source.attrs["z"] == 0.0
        assert np.sum(weights) == 1.0
        assert np.array_equal(stf_values, 7.0 + stf_time)

    preprocess_log = (tmp_path / "log/preprocess.log").read_text()
    assert "unified run" not in preprocess_log
    assert "falling back" not in preprocess_log
    assert "run" not in invoked_subcommands
