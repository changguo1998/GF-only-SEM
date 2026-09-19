"""Opt-in end-to-end regression for the compile-time C++ preprocess path."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from examples.halfspace.mesh_gen import create_regular_hex_mesh
from preprocess.pml_cpml import compute_cpml_parameters
from tools.gmsh_to_hdf5 import extract_topology, write_topology

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_CPP_SMOKE = os.environ.get("GF_RUN_CPP_PREPROCESS_SMOKE") == "1"


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _global_exchange_nodes(partition: h5py.File, neighbor: int) -> list[int]:
    global_nodes = np.asarray(partition["field/cell/local_cell2global_node"]).ravel()
    compact_nodes = np.asarray(partition["field/cell/local_cell2rank_node"]).ravel()
    compact_to_global = {
        int(compact): int(global_id) for global_id, compact in zip(global_nodes, compact_nodes)
    }
    dofs = np.asarray(partition[f"partition/exchange/neighbor_{neighbor}/send_dof"])
    return [compact_to_global[int(dof) // 3] for dof in dofs[::3]]


def _assert_uncompressed(group: h5py.Group | h5py.File) -> None:
    def check_dataset(_name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Dataset):
            assert item.compression is None

    group.visititems(check_dataset)


@pytest.mark.skipif(not RUN_CPP_SMOKE, reason="set GF_RUN_CPP_PREPROCESS_SMOKE=1")
def test_cpp_run_writes_solver_ready_partitions(tmp_path: Path) -> None:
    """C++ run should feed a two-rank MPI forward solve without Python preprocessing."""
    mpirun = shutil.which("mpirun")
    solver = PROJECT_ROOT / "bin/gf_solver_elastic_mpi"
    if mpirun is None:
        pytest.skip("mpirun is unavailable")
    if not solver.is_file():
        pytest.skip("gf_solver_elastic_mpi is not built")

    model_path = tmp_path / "model.h5"
    mesh = create_regular_hex_mesh(4, 4, 4, 4000.0, 4000.0, 4000.0)
    write_topology(str(model_path), extract_topology(mesh))

    build_dir = tmp_path / "build"
    runtime_dir = tmp_path / "bin"
    config_source = Path(__file__).with_name("config_user_cpp_smoke.cpp")
    _run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT / "preprocess/cpp"),
            "-B",
            str(build_dir),
            f"-DGF_USER_CONFIG={config_source}",
            f"-DGF_PREPROCESS_RUNTIME_DIR={runtime_dir}",
        ],
        cwd=tmp_path,
    )
    _run(["cmake", "--build", str(build_dir), "-j4"], cwd=tmp_path)

    preprocessor = runtime_dir / "gf_preprocess"
    _run(
        [
            str(preprocessor),
            "run",
            str(model_path),
            "--N",
            "2",
            "--cfl-safety",
            "0.5",
            "--nx",
            "4",
            "--ny",
            "4",
            "--n-ranks",
            "2",
            "--pml-xmin",
            "1",
            "--pml-xmax",
            "1",
            "--pml-ymin",
            "1",
            "--pml-ymax",
            "1",
            "--pml-zmin",
            "1",
            "--pml-zmax",
            "1",
        ],
        cwd=tmp_path,
    )

    with h5py.File(tmp_path / "config.h5", "r") as config:
        _assert_uncompressed(config)
        assert config["simulation"].attrs["nsteps"] == 100
        assert config["source/stf_t"].shape == (100,)
        assert config["source/cells/weights"].shape == (8, 27)
        assert np.sum(config["source/cells/weights"]) == pytest.approx(1.0)

        with h5py.File(model_path, "r") as model:
            domain = {name: float(config["domain"].attrs[name]) for name in config["domain"].attrs}
            solver_dt = float(config["simulation"].attrs["solver_dt"])
            reference = compute_cpml_parameters(
                np.asarray(model["field/element/coords"]),
                np.asarray(model["field/element/is_pml"], dtype=bool),
                domain,
                {
                    "xmin": 1000.0,
                    "xmax": 1000.0,
                    "ymin": 1000.0,
                    "ymax": 1000.0,
                    "zmin": 1000.0,
                    "zmax": 1000.0,
                },
                np.asarray(model["field/element/vp"]),
                f0_hz=5.0,
                dt=solver_dt,
            )
            dataset_names = {
                "pml_K": "cpml_K",
                "pml_d": "cpml_d",
                "pml_alpha": "cpml_alpha",
                "pml_coef_alpha": "pml_coef_alpha",
                "pml_coef_beta": "pml_coef_beta",
                "pml_coef_abar": "pml_coef_abar",
                "pml_coef_strain": "pml_coef_strain",
            }
            assert set(np.unique(reference["pml_region"])) == set(range(8))
            np.testing.assert_array_equal(
                model["field/element/pml_region"], reference["pml_region"]
            )
            for reference_name, model_name in dataset_names.items():
                np.testing.assert_allclose(
                    model[f"field/element/{model_name}"],
                    reference[reference_name],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            xmin_cell = 20
            xmin_coordinates = np.asarray(model["field/element/coords"])[xmin_cell, ..., 0].ravel()
            xmin_damping = np.asarray(model["field/element/cpml_d"])[xmin_cell, ..., 0]
            assert np.max(xmin_damping[xmin_coordinates == np.min(xmin_coordinates)]) > 0.0
            assert (
                np.max(np.abs(xmin_damping[xmin_coordinates == np.max(xmin_coordinates)]))
                < 1.0e-12
            )

    partition_paths = [tmp_path / f"partitions/partition_{rank}.h5" for rank in range(2)]
    partitions = [h5py.File(path, "r") for path in partition_paths]
    model = h5py.File(model_path, "r")
    try:
        required = (
            "field/cell/coords",
            "field/cell/local_cell2rank_node",
            "field/cell/local_cell2global_node",
            "field/cell/pml_coef_alpha",
            "field/cell/pml_coef_beta",
            "field/cell/pml_coef_abar",
            "field/cell/pml_coef_strain",
            "partition/local_cell_ids",
        )
        for partition in partitions:
            _assert_uncompressed(partition)
            assert all(name in partition for name in required)
            local_ids = np.asarray(partition["partition/local_cell_ids"])
            assert np.array_equal(
                partition["field/cell/coords"], model["field/cell/coords"][local_ids]
            )
            for coefficient_name in (
                "pml_coef_alpha",
                "pml_coef_beta",
                "pml_coef_abar",
                "pml_coef_strain",
            ):
                np.testing.assert_allclose(
                    partition[f"field/cell/{coefficient_name}"],
                    model[f"field/cell/{coefficient_name}"][local_ids],
                    rtol=0.0,
                    atol=0.0,
                )
        assert _global_exchange_nodes(partitions[0], 1) == _global_exchange_nodes(partitions[1], 0)
        assert sum("recording" in partition for partition in partitions) > 0
    finally:
        model.close()
        for partition in partitions:
            partition.close()

    (tmp_path / "wavefields/x").mkdir(parents=True)
    _run([mpirun, "-n", "2", str(solver), "--direction", "x"], cwd=tmp_path)
    record_paths = sorted((tmp_path / "wavefields/x").glob("record_*.h5"))
    assert record_paths
    with h5py.File(record_paths[-1], "r") as record:
        assert record["strain"].shape[-1] == 6
        assert np.all(np.isfinite(record["strain"]))
        assert np.any(np.asarray(record["strain"]) != 0.0)
