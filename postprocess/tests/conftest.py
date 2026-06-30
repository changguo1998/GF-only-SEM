"""Shared fixtures for postprocess tests."""

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def synthetic_model_path(tmp_path):
    """Create a minimal model.h5 with vertex coordinates."""
    path = tmp_path / "model.h5"
    n_vertex = 8  # unit cube corners

    with h5py.File(path, "w") as f:
        # Topology
        topo = f.create_group("topology")
        v2c = np.array(
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
        topo.create_dataset("vertex_to_coord", data=v2c)

        # Domain
        domain = f.create_group("domain")
        domain.attrs["xmin"] = 0.0
        domain.attrs["xmax"] = 1.0
        domain.attrs["ymin"] = 0.0
        domain.attrs["ymax"] = 1.0
        domain.attrs["zmin"] = 0.0
        domain.attrs["zmax"] = 1.0

    return path


@pytest.fixture
def synthetic_config_path(tmp_path):
    """Create a minimal config.h5 with simulation attrs."""
    path = tmp_path / "config.h5"
    with h5py.File(path, "w") as f:
        sim = f.create_group("simulation")
        sim.attrs["solver_dt"] = 0.01
        sim.attrs["output_dt_s"] = 0.01
        sim.attrs["nsteps"] = 2
        sim.attrs["nx_elements"] = 16
        sim.attrs["ny_elements"] = 16
        sim.attrs["nz_elements"] = 8
        sim.attrs["pml_xmin"] = 3
        sim.attrs["pml_xmax"] = 3
        sim.attrs["pml_ymin"] = 3
        sim.attrs["pml_ymax"] = 3
        sim.attrs["pml_zmin"] = 0
        sim.attrs["pml_zmax"] = 3
        sim.create_dataset("tilex_elements", data=np.array([5, 5], dtype=np.int64))
        sim.create_dataset("tiley_elements", data=np.array([5, 5], dtype=np.int64))
        sim.attrs["record_depth_max_m"] = 1.0
        sim.attrs["record_depth_actual_m"] = 1.0
    return path


@pytest.fixture
def synthetic_record_path(tmp_path):
    """Create a minimal single-rank record.h5 with 2 snapshots, 1 vertex."""
    n_checkpoints = 2
    n_vertices = 1
    path = tmp_path / "record_0.h5"
    with h5py.File(path, "w") as f:
        f.attrs["source_direction"] = "x"
        f.attrs["basis"] = "mesh_vertices"
        f.attrs["excludes_pml"] = True
        f.create_dataset("vertex_ids", data=np.array([1], dtype=np.int64))
        strain = np.zeros((n_checkpoints, n_vertices, 6), dtype=np.float64)
        for t in range(n_checkpoints):
            strain[t, 0, :] = [
                float(t) + 1.0,
                2.0 * (float(t) + 1.0),
                3.0 * (float(t) + 1.0),
                0.0,
                0.0,
                0.0,
            ]
        f.create_dataset("strain", data=strain, maxshape=(None, n_vertices, 6))
    return path


@pytest.fixture
def synthetic_multirank_records(tmp_path, synthetic_model_path):
    """Create per-rank record files for a 2-rank case, each with 1 vertex."""
    n_checkpoints = 2
    rank_list = [0, 1]
    paths = []
    for rank in rank_list:
        path = tmp_path / f"record_{rank}.h5"
        paths.append(path)
        with h5py.File(path, "w") as f:
            f.attrs["source_direction"] = "x"
            f.attrs["basis"] = "mesh_vertices"
            f.attrs["excludes_pml"] = True
            f.create_dataset("vertex_ids", data=np.array([rank + 1], dtype=np.int64))
            strain = np.zeros((n_checkpoints, 1, 6), dtype=np.float64)
            for t in range(n_checkpoints):
                val = float(t) + 1.0 + float(rank) * 10.0
                strain[t, 0, :] = [val, val, val, 0.0, 0.0, 0.0]
            f.create_dataset("strain", data=strain, maxshape=(None, 1, 6))
    return paths
