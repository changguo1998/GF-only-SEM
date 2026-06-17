"""CLI entry point for the preprocessor.

Usage:
    python -m preprocess mesh.h5 config.py
"""

import argparse
import os
import sys
import time

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

    # Step: PML damping
    print(f"[preprocess] Computing PML damping profiles...")
    try:
        from preprocess.pml import compute_pml_damping
        damping = compute_pml_damping(topology, coords, config.pml_thickness,
                                       domain_bounds)
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

    # Step: STF evaluation
    print(f"[preprocess] Evaluating STF...")
    try:
        from preprocess.stf_evaluator import evaluate_stf
        stf_t, stf_values = evaluate_stf(config.stf_func,
                                          config.output_dt,
                                          config.nsteps)
    except ImportError:
        stf_t = np.arange(config.nsteps) * config.output_dt
        stf_values = np.array([config.stf_func(t) for t in stf_t])

    # Source position
    source_x = float(config.source_x)
    source_y = float(config.source_y)
    source_z = float(domain_bounds["zmin"])  # auto on free surface
    source_xyz = np.array([source_x, source_y, source_z])

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
    write_config(config_h5, config, domain_bounds, stf_t, stf_values,
                 source_xyz)

    elapsed = time.time() - start
    print(f"[preprocess] Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()