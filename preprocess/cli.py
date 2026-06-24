"""CLI entry point for the preprocessor.

Usage:
    python -m preprocess mesh.h5 config.py
"""

import argparse
import os
import sys
import time
import math

import numpy as np

from preprocess.config_loader import load_config, ConfigValidationError
from preprocess.topology_reader import read_topology
from preprocess.gll_geometry import compute_gll_geometry
from preprocess.model_loader import load_and_interpolate


def main(argv: list[str] | None = None) -> None:
    """Run the full preprocessor pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess mesh.h5 + config.py → mesh.h5 + partitions/*.h5 + configs/config.h5"
    )
    parser.add_argument("mesh", help="Path to mesh.h5 (converter output)")
    parser.add_argument("config", help="Path to config.py (Python config script)")
    args = parser.parse_args(argv)

    start = time.time()
    mesh_path = os.path.abspath(args.mesh)
    config_path = os.path.abspath(args.config)

    print(f"[preprocess] Loading config: {config_path}")
    config = load_config(config_path)

    print(f"[preprocess] Reading topology from: {mesh_path}")
    topology = read_topology(mesh_path)
    n_cell = topology.n_cell
    n_surface = topology.n_surface
    n_vertex = topology.n_vertex

    # Auto-detect domain bounds from vertices
    v2c = topology.vertex_to_coord
    domain_bounds = {
        "xmin": float(v2c[:, 0].min()),
        "xmax": float(v2c[:, 0].max()),
        "ymin": float(v2c[:, 1].min()),
        "ymax": float(v2c[:, 1].max()),
        "zmin": float(v2c[:, 2].min()),
        "zmax": float(v2c[:, 2].max()),
    }
    print(f"[preprocess] Domain bounds: {domain_bounds}")

    N = int(config.polynomial_order)
    n_gll = N + 1

    # Step: compute GLL geometry
    print(f"[preprocess] Computing GLL geometry (N={N}, {n_cell} cells)...")
    coords, jacobian, dxi_dx, mass_gll = compute_gll_geometry(topology, N)

    # Step: load/interpolate material
    model_path = getattr(config, "model_path", None)
    print(f"[preprocess] Loading material model...")
    vp_gll, vs_gll, density_gll = load_and_interpolate(model_path, coords)

    # Step: boundary detection
    print(f"[preprocess] Detecting boundaries...")
    from preprocess.boundary_detector import detect_boundaries
    boundary_tag, is_pml = detect_boundaries(topology, domain_bounds)

    # Step: CFL validation — compute solver_dt and snapshot_stride
    print(f"[preprocess] Running CFL validation...")
    from preprocess.cfl_validator import compute_cfl_dt, compute_solver_dt
    cfl_dt = compute_cfl_dt(coords, vp_gll, config.cfl_safety)

    # Derive solver timestep and snapshot stride from output_dt_s
    solver_dt, snapshot_stride = compute_solver_dt(config.output_dt_s, cfl_dt)
    print(f"[preprocess]   cfl_dt={cfl_dt:.6e}, solver_dt={solver_dt:.6e}, snapshot_stride={snapshot_stride}")

    # Derive nsteps from total_duration_s
    nsteps = math.ceil(config.total_duration_s / solver_dt)
    total_duration_actual = nsteps * solver_dt
    if abs(total_duration_actual - config.total_duration_s) > 1e-12:
        print(f"[preprocess]   Adjusted total_duration_s from {config.total_duration_s} to {total_duration_actual}")

    # Step: source location
    print(f"[preprocess] Locating source on free surface...")
    from preprocess.source_locator import locate_source
    source_z = float(domain_bounds["zmin"])
    source_x_m = float(config.source_x_m)
    source_y_m = float(config.source_y_m)
    source_xyz_arr = np.array([source_x_m, source_y_m, source_z], dtype=np.float64)
    src_result = locate_source(topology, source_xyz_arr, coords, boundary_tag, N)
    print(f"[preprocess]   Source in {src_result['n_src_elem']} element(s)")

    # Step: STF evaluation (uses solver_dt, not output_dt_s)
    print(f"[preprocess] Evaluating STF...")
    try:
        from preprocess.stf_evaluator import evaluate_stf
        stf_t, stf_values = evaluate_stf(config.stf_func,
                                          solver_dt,
                                          nsteps)
    except ImportError:
        stf_t = np.arange(nsteps) * solver_dt
        stf_values = np.array([config.stf_func(t) for t in stf_t])

    # Step: pre-flight validation
    print(f"[preprocess] Running pre-flight validation...")
    from preprocess.preflight import run_preflight, PreflightError
    try:
        strict = getattr(config, "strict_validation", True)
        preflight_result = run_preflight(
            topology, coords, jacobian, vp_gll, vs_gll, density_gll,
            boundary_tag, domain_bounds, config, source_xyz_arr,
            stf_values, cfl_dt, nsteps, snapshot_stride, n_gll,
            strict=strict,
        )
        print(f"[preprocess]   {preflight_result.report()}")
    except PreflightError as e:
        print(f"[preprocess] PREFLIGHT FAILED:\n{e}")
        sys.exit(1)

    # Step: PML damping
    print(f"[preprocess] Computing PML damping profiles...")
    try:
        from preprocess.pml import compute_pml_damping
        damping = compute_pml_damping(topology, coords, config.pml_thickness,
                                       domain_bounds, is_pml)
    except ImportError:
        damping = np.zeros((n_cell, n_gll, n_gll, n_gll), dtype=np.float64)
        print(f"[preprocess]   pml.py not available — damping = 0")

    # Step: partition
    n_ranks = int(config.n_ranks)
    print(f"[preprocess] Partitioning into {n_ranks} ranks...")
    try:
        from preprocess.partition import partition
        partition_result = partition(topology, coords, n_ranks)
    except ImportError:
        partition_result = None
        print(f"[preprocess]   partition.py not available — skipping")

    # Step: write outputs
    fields = {
        "coords": coords,
        "jacobian": jacobian,
        "dxi_dx": dxi_dx,
        "mass": mass_gll,
        "vp": vp_gll,
        "vs": vs_gll,
        "density": density_gll,
        "is_pml": is_pml,
        "damping": damping,
    }

    print(f"[preprocess] Writing model to: {mesh_path}")
    from preprocess.model_writer import write_model
    write_model(mesh_path, topology, fields, boundary_tag, domain_bounds,
                partition_result)

    config_h5 = os.path.join(os.path.dirname(mesh_path), "configs", "config.h5")
    print(f"[preprocess] Writing config to: {config_h5}")
    from preprocess.config_writer import write_config
    write_config(
        config_h5, config, domain_bounds, stf_t, stf_values,
        source_xyz_arr, source_loc_result=src_result,
        solver_dt=solver_dt,
        snapshot_stride=snapshot_stride,
        nsteps=nsteps,
    )

    elapsed = time.time() - start
    print(f"[preprocess] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()