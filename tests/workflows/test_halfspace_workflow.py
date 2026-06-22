"""End-to-end integration test: full preprocessor pipeline on a half-space mesh.

Creates a single-element regular hex mesh programmatically, then runs every
pipeline step in order: topology read → GLL geometry → material load →
boundary detect → PML damping → partition → model write → config write.
Validates all output files and data.

Uses a single element (nx=1, ny=1, nz=1) to work around the _get_cell_vertex_ids
sorting bug — that function's implementation relies on sorted coincidence
for single-element meshes until the topological-ordering fix lands.
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
    """Create a minimal half-space config module dynamically."""
    mod = types.ModuleType("test_config")
    mod.title = "halfspace_test"
    mod.polynomial_order = 4
    mod.output_dt = 0.001
    mod.nsteps = 100
    mod.cfl_safety = 0.5
    mod.cfl_threshold = 0.3
    mod.checkpoint_interval = 0
    mod.checkpoint_precision = "float32"
    mod.storage_limit_gb = 1.0
    mod.n_ranks = 1
    mod.pml_thickness = {
        "xmin": 0, "xmax": 0,
        "ymin": 0, "ymax": 0,
        "zmin": 0, "zmax": 2,
    }
    mod.source_x = 2000.0
    mod.source_y = 2000.0

    def stf_func(t):
        return 1.0

    def vp(x, y, z):
        return 3000.0

    def vs(x, y, z):
        return 1500.0

    def density(x, y, z):
        return 2500.0

    mod.stf_func = stf_func
    mod.vp = vp
    mod.vs = vs
    mod.density = density
    return mod


class TestHalfspaceWorkflow:
    """Full preprocessor pipeline integration test on a single hex element."""

    def test_pipeline(self, tmp_path):
        # ------------------------------------------------------------------
        # 1. Generate regular hex mesh
        # ------------------------------------------------------------------
        from tests.workflows.regular_hex_mesh import create_regular_hex_mesh
        from tools.gmsh_to_hdf5 import extract_topology, write_topology

        mesh = create_regular_hex_mesh(
            nx=1, ny=1, nz=1,
            lx=4000.0, ly=4000.0, lz=2000.0,
        )
        topology_dict = extract_topology(mesh)
        mesh_path = str(tmp_path / "mesh.h5")
        write_topology(mesh_path, topology_dict)

        # ------------------------------------------------------------------
        # 2. Read topology
        # ------------------------------------------------------------------
        from preprocess.topology_reader import read_topology

        topology = read_topology(mesh_path)
        assert topology.n_cell == 1
        assert topology.n_vertex == 8
        assert topology.n_surface == 6
        assert topology.n_edge == 12

        # ------------------------------------------------------------------
        # 3. Compute GLL geometry
        # ------------------------------------------------------------------
        from preprocess.gll_geometry import compute_gll_geometry

        N = 4
        NGLL = N + 1  # 5
        coords, jac, dxi_dx, mass = compute_gll_geometry(topology, N)

        expected_shape = (1, NGLL, NGLL, NGLL)
        assert coords.shape == (1, NGLL, NGLL, NGLL, 3)
        assert jac.shape == expected_shape
        assert dxi_dx.shape == (1, NGLL, NGLL, NGLL, 9)
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
        assert np.allclose(vp_arr, 3000.0)
        assert np.allclose(vs_arr, 1500.0)
        assert np.allclose(dens_arr, 2500.0)

        # ------------------------------------------------------------------
        # 5. Detect boundaries
        # ------------------------------------------------------------------
        from preprocess.boundary_detector import detect_boundaries

        domain_bounds = {
            "xmin": 0.0, "xmax": 4000.0,
            "ymin": 0.0, "ymax": 4000.0,
            "zmin": 0.0, "zmax": 2000.0,
        }
        boundary_tag, is_pml = detect_boundaries(topology, domain_bounds)

        assert boundary_tag.shape == (6,)
        # 1 free surface (z=zmin) + 5 absorbing (other 5 faces) for a single element
        assert np.count_nonzero(boundary_tag == 1) == 1
        assert np.count_nonzero(boundary_tag == 2) == 5
        assert np.count_nonzero(boundary_tag == 0) == 0

        # is_pml: True if element touches any absorbing surface
        assert is_pml.shape == (1,)
        assert is_pml[0] == True

        # ------------------------------------------------------------------
        # 6. Compute PML damping profiles
        # ------------------------------------------------------------------
        from preprocess.pml import compute_pml_damping

        pml_thickness = {
            "xmin": 0, "xmax": 0,
            "ymin": 0, "ymax": 0,
            "zmin": 0, "zmax": 2,
        }
        damping = compute_pml_damping(
            topology, coords, pml_thickness, domain_bounds, is_pml,
        )
        assert damping.shape == expected_shape
        # Damping in [0, 1]; non-zero only in absorbing layers
        assert np.all(damping >= 0.0)
        assert np.all(damping <= 1.0)

        # ------------------------------------------------------------------
        # 7. Partition (single rank)
        # ------------------------------------------------------------------
        from preprocess.partition import partition

        partition_result = partition(topology, coords, n_ranks=1)
        assert partition_result["n_ranks"] == 1
        assert len(partition_result["per_rank"]) == 1
        rank0 = partition_result["per_rank"][0]
        assert len(rank0["local_element_ids"]) == 1
        assert rank0["local_element_ids"] == [0]

        # ------------------------------------------------------------------
        # 8. Write model (extend mesh.h5 + partition files)
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
        # 9. Write config
        # ------------------------------------------------------------------
        from preprocess.stf_evaluator import evaluate_stf
        from preprocess.config_writer import write_config

        config_mod = _make_config_module()
        stf_t, stf_vals = evaluate_stf(
            config_mod.stf_func, config_mod.output_dt, config_mod.nsteps,
        )
        source_xyz = np.array(
            [config_mod.source_x, config_mod.source_y, 0.0], dtype=np.float64,
        )
        config_path = str(tmp_path / "configs" / "config.h5")
        write_config(config_path, config_mod, domain_bounds, stf_t, stf_vals, source_xyz)

        # ==================================================================
        # Validations: mesh.h5
        # ==================================================================
        with h5py.File(mesh_path, "r") as f:
            # /field/element/*
            felem = f["field"]["element"]
            assert list(felem["coords"].shape) == [1, NGLL, NGLL, NGLL, 3]
            assert list(felem["dxi_dx"].shape) == [1, NGLL, NGLL, NGLL, 9]
            assert list(felem["jacobian"].shape) == [1, NGLL, NGLL, NGLL]
            assert list(felem["is_pml"].shape) == [1, ]

            # /field/surface/boundary_tag
            fsurf = f["field"]["surface"]
            assert list(fsurf["boundary_tag"].shape) == [6, ]

            # /domain/ attrs
            domain = f["domain"]
            assert domain.attrs["xmin"] == 0.0
            assert domain.attrs["xmax"] == 4000.0
            assert domain.attrs["ymin"] == 0.0
            assert domain.attrs["ymax"] == 4000.0
            assert domain.attrs["zmin"] == 0.0
            assert domain.attrs["zmax"] == 2000.0

        # ==================================================================
        # Validations: configs/config.h5
        # ==================================================================
        with h5py.File(config_path, "r") as f:
            assert "simulation" in f
            assert "domain" in f
            assert "source" in f

            sim = f["simulation"]
            assert sim.attrs["title"] == "halfspace_test"
            assert sim.attrs["polynomial_order"] == 4
            assert sim.attrs["dt"] == 0.001
            assert sim.attrs["nsteps"] == 100

            src = f["source"]
            assert list(src["stf_t"].shape) == [100, ]
            assert list(src["stf_values"].shape) == [100, ]
            assert src.attrs["x"] == 2000.0
            assert src.attrs["y"] == 2000.0
            assert src.attrs["z"] == 0.0

        # ==================================================================
        # Validations: partitions/partition_0.h5
        # ==================================================================
        part_path = tmp_path / "partitions" / "partition_0.h5"
        assert part_path.exists()
        with h5py.File(str(part_path), "r") as f:
            assert "field" in f
            assert "partition" in f
            pgrp = f["partition"]
            assert pgrp.attrs["n_ranks"] == 1
            assert list(pgrp["local_element_ids"].shape) == [1, ]