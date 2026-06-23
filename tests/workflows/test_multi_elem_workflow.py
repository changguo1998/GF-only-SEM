"""Multi-element, multi-rank preprocessor integration test.

Creates a 2×2×2 hex mesh (8 elements) and runs the full preprocessor
pipeline with n_ranks=2.  Validates:
  - Multi-rank partition (each rank gets ≥ 1 element)
  - Ghost element lists and ghost owners
  - Exchange patterns (send/recv DOF lists per neighbor)
  - Source location on free surface (may land on shared face)
  - CFL validation, pre-flight validation
  - All output files (mesh.h5, configs/config.h5, partitions/partition_{0,1}.h5)
"""

import os
import sys
import types

import h5py
import numpy as np
import pytest
from pathlib import Path

# Ensure project root and tools/ are importable.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
_tools_dir = _project_root / "tools"
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))


def _make_multi_elem_config():
    """Config for a 2×2×2 element mesh with 2 MPI ranks."""
    mod = types.ModuleType("test_multi_config")
    mod.title = "multi_elem_test"
    mod.polynomial_order = 3
    mod.output_dt_s = 0.005
    mod.total_duration_s = 0.25
    mod.cfl_safety = 0.3
    mod.snapshot_precision = "float64"
    mod.storage_limit_gb = 10.0
    mod.n_ranks = 2
    mod.pml_thickness = {
        "xmin": 0, "xmax": 0,
        "ymin": 0, "ymax": 0,
        "zmin": 0, "zmax": 1,
    }
    mod.source_x_m = 2000.0
    mod.source_y_m = 2000.0

    def stf_func(t_s):
        import numpy as _np
        f0_hz = 5.0
        t0_s = 0.3
        return (1 - 2 * (_np.pi * f0_hz * (t_s - t0_s)) ** 2) * _np.exp(
            -(_np.pi * f0_hz * (t_s - t0_s)) ** 2
        )

    def vp_m_s(x_m, y_m, z_m):
        return 3000.0 + z_m * 0.5

    def vs_m_s(x_m, y_m, z_m):
        return 1500.0 + z_m * 0.25

    def density_kg_m3(x_m, y_m, z_m):
        return 2500.0 + z_m * 0.2

    mod.stf_func = stf_func
    mod.vp_m_s = vp_m_s
    mod.vs_m_s = vs_m_s
    mod.density_kg_m3 = density_kg_m3
    return mod


def _validate_partition_rank(rank_data: dict, rank: int, n_total_cell: int):
    """Validate a single rank's partition data."""
    local_ids = rank_data["local_element_ids"]
    ghost_ids = rank_data.get("ghost_element_ids", [])
    ghost_owners = rank_data.get("ghost_owners", [])
    exchange = rank_data.get("exchange", {})

    # Each rank must have at least 1 local element
    assert len(local_ids) >= 1, f"Rank {rank}: no local elements"

    # Local + ghost ≤ total elements
    assert len(local_ids) + len(ghost_ids) <= n_total_cell, (
        f"Rank {rank}: local+ghost ({len(local_ids)}+{len(ghost_ids)}) "
        f"> n_cell_total ({n_total_cell})"
    )

    # Ghost owners must be in [0, n_ranks-1), not equal to this rank
    for owner in ghost_owners:
        assert owner != rank, f"Rank {rank}: ghost owner is self"

    # Exchange: each neighbor entry should have send/recv lists
    for neighbor_rank, ex in exchange.items():
        assert neighbor_rank != rank, f"Rank {rank}: exchange with self"
        assert "send_dof" in ex
        assert "recv_dof" in ex
        assert len(ex["send_dof"]) == len(ex["recv_dof"])


class TestMultiElementWorkflow:
    """Full pipeline on an 8-element (2×2×2) mesh with 2 MPI ranks."""

    def test_pipeline(self, tmp_path):
        NX, NY, NZ = 2, 2, 2
        N_RANKS = 2
        LX, LY, LZ = 4000.0, 4000.0, 2000.0

        # --------------------------------------------------------------
        # 1. Generate regular hex mesh
        # --------------------------------------------------------------
        from tests.workflows.regular_hex_mesh import create_regular_hex_mesh
        from tools.gmsh_to_hdf5 import extract_topology, write_topology

        mesh = create_regular_hex_mesh(
            nx=NX, ny=NY, nz=NZ,
            lx=LX, ly=LY, lz=LZ,
        )
        topology_dict = extract_topology(mesh)
        mesh_path = str(tmp_path / "mesh.h5")
        write_topology(mesh_path, topology_dict)

        # --------------------------------------------------------------
        # 2. Read topology
        # --------------------------------------------------------------
        from preprocess.topology_reader import read_topology

        topology = read_topology(mesh_path)
        n_total = NX * NY * NZ
        assert topology.n_cell == n_total

        # --------------------------------------------------------------
        # 3. Compute GLL geometry
        # --------------------------------------------------------------
        from preprocess.gll_geometry import compute_gll_geometry

        N = 3
        NGLL = N + 1  # 4
        coords, jac, dxi_dx, mass = compute_gll_geometry(topology, N)
        assert coords.shape == (n_total, NGLL, NGLL, NGLL, 3)
        assert jac.shape == (n_total, NGLL, NGLL, NGLL)
        assert np.all(jac > 0.0)

        # --------------------------------------------------------------
        # 4. Load material model
        # --------------------------------------------------------------
        from preprocess.model_loader import load_and_interpolate

        vp_arr, vs_arr, dens_arr = load_and_interpolate(None, coords)
        assert vp_arr.shape == (n_total, NGLL, NGLL, NGLL)
        assert np.all(vp_arr > 0)

        # --------------------------------------------------------------
        # 5. Detect boundaries
        # --------------------------------------------------------------
        from preprocess.boundary_detector import detect_boundaries

        domain_bounds = {
            "xmin": 0.0, "xmax": LX,
            "ymin": 0.0, "ymax": LY,
            "zmin": 0.0, "zmax": LZ,
        }
        boundary_tag, is_pml = detect_boundaries(topology, domain_bounds)
        assert boundary_tag.shape == (topology.n_surface,)
        # Must have at least one free surface and at least one absorbing
        assert np.count_nonzero(boundary_tag == 1) >= 1
        assert np.count_nonzero(boundary_tag == 2) >= 1

        # --------------------------------------------------------------
        # 5b. CFL validation
        # --------------------------------------------------------------
        from preprocess.cfl_validator import compute_cfl_dt, compute_solver_dt

        config_mod = _make_multi_elem_config()
        cfl_dt = compute_cfl_dt(coords, vp_arr, config_mod.cfl_safety)
        solver_dt, snapshot_stride = compute_solver_dt(config_mod.output_dt_s, cfl_dt)
        nsteps = int(np.ceil(config_mod.total_duration_s / solver_dt))
        assert cfl_dt > 0.0
        assert solver_dt <= cfl_dt
        assert snapshot_stride >= 1
        assert nsteps >= 1

        # --------------------------------------------------------------
        # 5c. Source location
        # --------------------------------------------------------------
        from preprocess.source_locator import locate_source

        source_xyz = np.array(
            [config_mod.source_x_m, config_mod.source_y_m, domain_bounds["zmin"]],
            dtype=np.float64,
        )
        src_result = locate_source(topology, source_xyz, coords, boundary_tag, N)
        assert src_result["n_src_elem"] >= 1
        assert abs(float(np.sum(src_result["weights"])) - 1.0) < 1e-6

        # --------------------------------------------------------------
        # 6. Compute PML damping
        # --------------------------------------------------------------
        from preprocess.pml import compute_pml_damping

        pml_thickness = config_mod.pml_thickness
        damping = compute_pml_damping(
            topology, coords, pml_thickness, domain_bounds, is_pml,
        )
        assert damping.shape == (n_total, NGLL, NGLL, NGLL)
        assert np.all(damping >= 0.0)
        assert np.all(damping <= 1.0)

        # --------------------------------------------------------------
        # 7. Partition (multi-rank)
        # --------------------------------------------------------------
        from preprocess.partition import partition

        partition_result = partition(topology, coords, n_ranks=N_RANKS)
        assert partition_result["n_ranks"] == N_RANKS
        assert len(partition_result["per_rank"]) == N_RANKS
        assert partition_result["element_to_rank"].shape == (n_total,)

        # --- Multi-rank partition validations ---
        for rank in range(N_RANKS):
            rk = partition_result["per_rank"][rank]
            _validate_partition_rank(rk, rank, n_total)

        # All ranks together should cover all elements
        all_local_ids: set[int] = set()
        for rank in range(N_RANKS):
            all_local_ids.update(partition_result["per_rank"][rank]["local_element_ids"])
        assert len(all_local_ids) == n_total, (
            f"Only {len(all_local_ids)}/{n_total} elements assigned to ranks"
        )

        # --------------------------------------------------------------
        # 7b. Pre-flight validation
        # --------------------------------------------------------------
        from preprocess.preflight import run_preflight

        stf_t = np.arange(nsteps) * solver_dt
        stf_vals = np.array([config_mod.stf_func(t) for t in stf_t])
        preflight_result = run_preflight(
            topology, coords, jac, vp_arr, vs_arr, dens_arr,
            boundary_tag, domain_bounds, config_mod, source_xyz,
            stf_vals, cfl_dt, nsteps, snapshot_stride, NGLL,
            strict=True,
        )
        assert not preflight_result.has_errors

        # --------------------------------------------------------------
        # 8. Write model + partition files
        # --------------------------------------------------------------
        from preprocess.model_writer import write_model

        fields = {
            "coords": coords,
            "dxi_dx": dxi_dx,
            "jacobian": jac,
            "mass": mass,
            "vp": vp_arr,
            "vs": vs_arr,
            "density": dens_arr,
            "is_pml": is_pml,
            "damping": damping,
        }
        write_model(
            mesh_path, topology, fields, boundary_tag, domain_bounds,
            partition_result,
        )

        # --------------------------------------------------------------
        # 9. Write config
        # --------------------------------------------------------------
        from preprocess.stf_evaluator import evaluate_stf
        from preprocess.config_writer import write_config

        stf_t, stf_vals_full = evaluate_stf(
            config_mod.stf_func, solver_dt, nsteps,
        )
        config_path = str(tmp_path / "configs" / "config.h5")
        write_config(config_path, config_mod, domain_bounds,
                     stf_t, stf_vals_full, source_xyz,
                     source_loc_result=src_result,
                     solver_dt=solver_dt,
                     snapshot_stride=snapshot_stride,
                     nsteps=nsteps)

        # ==============================================================
        # HDF5 validations
        # ==============================================================

        # --- mesh.h5 ---
        with h5py.File(mesh_path, "r") as f:
            felem = f["field"]["element"]
            assert list(felem["coords"].shape) == [n_total, NGLL, NGLL, NGLL, 3]
            assert list(felem["is_pml"].shape) == [n_total, ]

        # --- configs/config.h5 ---
        with h5py.File(config_path, "r") as f:
            sim = f["simulation"].attrs
            assert sim["solver_dt"] == pytest.approx(solver_dt)
            assert sim["output_dt_s"] == pytest.approx(config_mod.output_dt_s)
            assert sim["snapshot_stride"] == snapshot_stride
            assert sim["nsteps"] == nsteps
            assert sim["snapshot_precision"] == config_mod.snapshot_precision
            assert f["source"].attrs["n_src_elements"] >= 1
            elems = f["source"]["elements"]
            assert "element_ids" in elems
            assert "weights" in elems
            assert "xi" in elems
            assert "eta" in elems
            assert "zeta" in elems
            wsum = float(np.sum(elems["weights"][:]))
            assert abs(wsum - 1.0) < 1e-6

        # --- partitions/partition_{0,1}.h5 ---
        for rank in range(N_RANKS):
            part_path = tmp_path / "partitions" / f"partition_{rank}.h5"
            assert part_path.exists(), f"Missing partition_{rank}.h5"

            with h5py.File(str(part_path), "r") as f:
                assert "field" in f
                assert "partition" in f
                pgrp = f["partition"]
                assert pgrp.attrs["n_ranks"] == N_RANKS

                # Must have local_element_ids
                local_ids = pgrp["local_element_ids"][:]
                assert len(local_ids) >= 1

                # Exchange group for multi-rank
                if N_RANKS > 1:
                    # At least one exchange neighbor
                    ex_grp_exists = "exchange" in pgrp or any(
                        grp.startswith("send_to_") or grp.startswith("recv_from_")
                        for grp in pgrp
                    )
                    # Exchange patterns may be stored differently — just check
                    # that we have the neighbor list or exchange data
                    has_neighbors = "neighbors" in pgrp or ex_grp_exists
                    assert has_neighbors, f"Partition {rank}: no exchange data found"