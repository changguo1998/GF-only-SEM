"""End-to-end integration test: full preprocessor pipeline on a half-space mesh.

Creates a production-scale 500k-element regular hex mesh programmatically,
then runs every pipeline step in order: topology read → GLL geometry → material
load → boundary detect → PML damping → partition → model write → config write.
Validates all output files and data.

This test is slow/heavy by design. It is skipped unless GF_RUN_SLOW=1.
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


def _make_config_module():
    """Create a minimal half-space config module with new field names."""
    import math
    mod = types.ModuleType("test_config")
    mod.title = "halfspace_10x10x5"
    mod.polynomial_order = 4
    mod.output_dt_s = 0.01
    mod.total_duration_s = 5.0
    mod.cfl_safety = 0.5
    mod.snapshot_precision = "float32"
    mod.storage_limit_gb = 2500
    mod.n_ranks = 2
    mod.pml_thickness = {
        "xmin": 5, "xmax": 5,
        "ymin": 5, "ymax": 5,
        "zmin": 0, "zmax": 5,
    }
    mod.source_x_m = 5000.0
    mod.source_y_m = 5000.0

    def stf_func(t_s):
        import numpy as np
        f0_hz = 2.0
        t0_s = 1.0
        a = np.pi * f0_hz * (t_s - t0_s)
        return (1 - 2 * a**2) * np.exp(-a**2)

    def vp_m_s(x_m, y_m, z_m):
        return 5000.0

    def vs_m_s(x_m, y_m, z_m):
        return 3000.0

    def density_kg_m3(x_m, y_m, z_m):
        return 2700.0

    mod.stf_func = stf_func
    mod.vp_m_s = vp_m_s
    mod.vs_m_s = vs_m_s
    mod.density_kg_m3 = density_kg_m3
    return mod


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("GF_RUN_SLOW") != "1",
    reason="500k-element workflow; set GF_RUN_SLOW=1 to run",
)
class TestHalfspaceWorkflow:
    """Full preprocessor pipeline integration test on a 500k-element half-space mesh."""

    def test_pipeline(self, tmp_path):
        import math

        # ------------------------------------------------------------------
        # 1. Generate regular hex mesh
        # ------------------------------------------------------------------
        from tests.workflows.regular_hex_mesh import create_regular_hex_mesh
        from tools.gmsh_to_hdf5 import extract_topology, write_topology

        # 500k elements: 100x100x50, 10 km x 10 km x 5 km, 100 m element size
        mesh = create_regular_hex_mesh(
            nx=100, ny=100, nz=50,
            lx=10000.0, ly=10000.0, lz=5000.0,
        )
        topology_dict = extract_topology(mesh)
        mesh_path = str(tmp_path / "mesh.h5")
        write_topology(mesh_path, topology_dict)

        # ------------------------------------------------------------------
        # 2. Read topology
        # ------------------------------------------------------------------
        from preprocess.topology_reader import read_topology

        topology = read_topology(mesh_path)
        n_cell = topology.n_cell
        assert n_cell == 100 * 100 * 50
        assert topology.n_vertex == (100 + 1) * (100 + 1) * (50 + 1)

        # ------------------------------------------------------------------
        # 3. Compute GLL geometry
        # ------------------------------------------------------------------
        from preprocess.gll_geometry import compute_gll_geometry

        N = 4
        NGLL = N + 1  # 5
        coords, jac, dxi_dx, mass = compute_gll_geometry(topology, N)

        expected_shape = (n_cell, NGLL, NGLL, NGLL)
        assert coords.shape == (n_cell, NGLL, NGLL, NGLL, 3)
        assert jac.shape == expected_shape
        assert dxi_dx.shape == (n_cell, NGLL, NGLL, NGLL, 9)
        assert mass.shape == expected_shape

        # Jacobians must be positive (non-degenerate hex)
        assert np.all(jac > 0.0)

        # ------------------------------------------------------------------
        # 4. Load material model (placeholder: constant values)
        # ------------------------------------------------------------------
        from preprocess.model_loader import load_and_interpolate

        vp_arr, vs_arr, dens_arr = load_and_interpolate(None, coords)

        assert vp_arr.shape == expected_shape
        assert vs_arr.shape == expected_shape
        assert dens_arr.shape == expected_shape
        # model_loader returns placeholder values; verify shape and finite
        assert np.all(vp_arr > 0)
        assert np.all(vs_arr >= 0)
        assert np.all(dens_arr > 0)
        assert np.all(np.isfinite(vp_arr))
        assert np.all(np.isfinite(vs_arr))
        assert np.all(np.isfinite(dens_arr))

        # ------------------------------------------------------------------
        # 5. Detect boundaries
        # ------------------------------------------------------------------
        from preprocess.boundary_detector import detect_boundaries

        domain_bounds = {
            "xmin": 0.0, "xmax": 10000.0,
            "ymin": 0.0, "ymax": 10000.0,
            "zmin": 0.0, "zmax": 5000.0,
        }
        boundary_tag, is_pml = detect_boundaries(topology, domain_bounds)

        n_surface = topology.n_surface
        assert boundary_tag.shape == (n_surface,)
        # At least one free surface (z=zmin) and one absorbing
        assert np.count_nonzero(boundary_tag == 1) >= 1
        assert np.count_nonzero(boundary_tag == 2) >= 1

        # ------------------------------------------------------------------
        # 6. Compute PML damping
        # ------------------------------------------------------------------
        from preprocess.pml import compute_pml_damping

        pml_thickness = {
            "xmin": 1, "xmax": 1,
            "ymin": 1, "ymax": 1,
            "zmin": 0, "zmax": 1,
        }
        damping = compute_pml_damping(
            topology, coords, pml_thickness, domain_bounds, is_pml,
        )
        assert damping.shape == expected_shape
        assert np.all(damping >= 0.0)
        assert np.all(damping <= 1.0)

        # ------------------------------------------------------------------
        # 7. CFL — compute solver_dt, snapshot_stride, nsteps
        # ------------------------------------------------------------------
        from preprocess.cfl_validator import compute_cfl_dt, compute_solver_dt

        config_mod = _make_config_module()

        cfl_dt = compute_cfl_dt(coords, vp_arr, config_mod.cfl_safety)
        assert cfl_dt > 0

        solver_dt, snapshot_stride = compute_solver_dt(config_mod.output_dt_s, cfl_dt)
        assert solver_dt <= cfl_dt
        assert solver_dt > 0
        assert snapshot_stride >= 1
        # output_dt_s must be an integer multiple of solver_dt
        assert abs(config_mod.output_dt_s / solver_dt - snapshot_stride) < 1e-12

        nsteps = math.ceil(config_mod.total_duration_s / solver_dt)
        total_actual = nsteps * solver_dt
        assert total_actual >= config_mod.total_duration_s
        assert nsteps >= config_mod.total_duration_s / solver_dt
        # nsteps must be integer multiple of snapshot_stride (last step lands on snapshot)
        assert nsteps % snapshot_stride == 0

        # ------------------------------------------------------------------
        # 8. Partition (n_ranks=2)
        # ------------------------------------------------------------------
        from preprocess.partition import partition

        n_ranks = config_mod.n_ranks
        partition_result = partition(topology, coords, n_ranks)
        assert partition_result["n_ranks"] == n_ranks
        assert len(partition_result["per_rank"]) == n_ranks
        # Verify each element assigned to exactly one rank
        all_elements = []
        for r in range(n_ranks):
            local = partition_result["per_rank"][r]["local_element_ids"]
            all_elements.extend(local)
        all_elements.sort()
        expected_ids = list(range(n_cell))
        assert all_elements == expected_ids, "each element assigned exactly once"

        # ------------------------------------------------------------------
        # 9. Write model (extend mesh.h5 + partition files)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 10. Write config
        # ------------------------------------------------------------------
        from preprocess.stf_evaluator import evaluate_stf
        from preprocess.config_writer import write_config

        stf_t, stf_vals = evaluate_stf(
            config_mod.stf_func, solver_dt, nsteps,
        )
        source_xyz = np.array(
            [config_mod.source_x_m, config_mod.source_y_m, 0.0], dtype=np.float64,
        )
        config_path = str(tmp_path / "configs" / "config.h5")
        write_config(
            config_path, config_mod, domain_bounds, stf_t, stf_vals, source_xyz,
            solver_dt=solver_dt,
            snapshot_stride=snapshot_stride,
            nsteps=nsteps,
        )

        # ==================================================================
        # Validations: mesh.h5
        # ==================================================================
        with h5py.File(mesh_path, "r") as f:
            felem = f["field"]["element"]
            assert list(felem["coords"].shape) == [n_cell, NGLL, NGLL, NGLL, 3]
            assert list(felem["dxi_dx"].shape) == [n_cell, NGLL, NGLL, NGLL, 9]
            assert list(felem["jacobian"].shape) == [n_cell, NGLL, NGLL, NGLL]
            assert list(felem["is_pml"].shape) == [n_cell, ]

            # /field/surface/boundary_tag
            fsurf = f["field"]["surface"]
            assert list(fsurf["boundary_tag"].shape) == [n_surface, ]

            # /domain/ attrs
            domain = f["domain"]
            assert domain.attrs["xmin"] == 0.0
            assert domain.attrs["xmax"] == 10000.0
            assert domain.attrs["ymin"] == 0.0
            assert domain.attrs["ymax"] == 10000.0
            assert domain.attrs["zmin"] == 0.0
            assert domain.attrs["zmax"] == 5000.0

        # ==================================================================
        # Validations: configs/config.h5 — new schema
        # ==================================================================
        with h5py.File(config_path, "r") as f:
            assert "simulation" in f
            assert "domain" in f
            assert "source" in f

            sim = f["simulation"]
            assert sim.attrs["title"] == "halfspace_10x10x5"
            assert sim.attrs["polynomial_order"] == 4
            assert sim.attrs["solver_dt"] == pytest.approx(solver_dt)
            assert sim.attrs["output_dt_s"] == 0.01
            assert sim.attrs["snapshot_stride"] == snapshot_stride
            assert sim.attrs["nsteps"] == nsteps
            assert sim.attrs["cfl_safety"] == 0.5
            assert sim.attrs["snapshot_precision"] == "float32"
            assert sim.attrs["storage_limit_gb"] == 2500

            src = f["source"]
            assert list(src["stf_t"].shape) == [nsteps, ]
            assert list(src["stf_values"].shape) == [nsteps, ]
            assert src.attrs["x"] == 5000.0
            assert src.attrs["y"] == 5000.0
            assert src.attrs["z"] == 0.0

        # ==================================================================
        # Validations: partitions files all exist
        # ==================================================================
        for r in range(n_ranks):
            part_path = tmp_path / "partitions" / f"partition_{r}.h5"
            assert part_path.exists(), f"partition_{r}.h5 not found"
            with h5py.File(str(part_path), "r") as f:
                assert "field" in f
                assert "partition" in f
                pgrp = f["partition"]
                assert pgrp.attrs["n_ranks"] == n_ranks
                n_local = len(partition_result["per_rank"][r]["local_element_ids"])
                assert list(pgrp["local_element_ids"].shape) == [n_local, ]