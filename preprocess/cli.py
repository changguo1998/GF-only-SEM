"""Preprocessor entry point.

Reads mesh.h5 + config.py from the current working directory.
"""

import math
import os
import sys
import time

import numpy as np
import logging

from preprocess.accelerator import run_accelerator
from preprocess.config_loader import load_config
from preprocess.gll_geometry import compute_gll_geometry
from preprocess.model_loader import load_and_interpolate
from preprocess.topology_reader import read_topology


def setup_logging(log_dir: str = "log") -> logging.Logger:
    """Create logger writing to log/preprocess.log with timestamps."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "preprocess.log")

    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.DEBUG)

    # File handler — detailed
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)

    # Console handler — info and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[preprocess] %(message)s"))
    logger.addHandler(ch)

    logger.info(f"Log file: {os.path.abspath(log_path)}")
    return logger


def main() -> None:
    """Run the full preprocessor pipeline using mesh.h5 + config.py in CWD."""
    logger = setup_logging()
    start = time.time()

    mesh_path = os.path.abspath("mesh.h5")
    config_path = os.path.abspath("config.py")

    logger.info(f"Loading config: {config_path}")
    config = load_config(config_path)
    logger.debug(f"  title={config.title}, N={config.polynomial_order}")
    logger.debug(f"  output_dt_s={config.output_dt_s}, total_duration_s={config.total_duration_s}")
    logger.debug(f"  cfl_safety={config.cfl_safety}, n_ranks={config.n_ranks}")
    logger.debug(
        f"  snapshot_precision={config.snapshot_precision}, storage_limit_gb={config.storage_limit_gb}"
    )

    logger.info(f"Reading topology from: {mesh_path}")
    topology = read_topology(mesh_path)
    n_cell = topology.n_cell
    n_surface = topology.n_surface
    n_vertex = topology.n_vertex
    logger.debug(f"  n_cell={n_cell}, n_surface={n_surface}, n_vertex={n_vertex}")

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
    logger.info(f"Domain bounds: {domain_bounds}")

    N = int(config.polynomial_order)
    n_gll = N + 1

    # Try C++ accelerator for GLL geometry, CFL, and PML damping
    accel_result = run_accelerator(mesh_path, config, domain_bounds)

    if accel_result["used_cpp"]:
        logger.info("Using C++-accelerated GLL geometry")
        coords = accel_result["coords"]
        jacobian = accel_result["jacobian"]
        dxi_dx = accel_result["dxi_dx"]
        mass_gll = accel_result["mass"]
        damping_cpp = accel_result["damping"]
        cfl_dt = accel_result["cfl_dt"]
    else:
        # Step: compute GLL geometry (Python fallback)
        logger.info(f"Computing GLL geometry (N={N}, {n_cell} cells)...")
        t0 = time.time()
        coords, jacobian, dxi_dx, mass_gll = compute_gll_geometry(topology, N)
        logger.debug(f"  GLL geometry: {time.time() - t0:.2f}s")
        damping_cpp = None

    # Step: load/interpolate material (always Python — user callables)
    model_path = getattr(config, "model_path", None)
    logger.info("Loading material model...")
    t0 = time.time()
    vp_gll, vs_gll, density_gll = load_and_interpolate(model_path, coords, config=config)
    logger.debug(f"  material model: {time.time() - t0:.2f}s")

    # Step: boundary detection
    logger.info("Detecting boundaries...")
    from preprocess.boundary_detector import detect_boundaries

    boundary_tag, is_pml = detect_boundaries(topology, domain_bounds)

    # Step: CFL validation — compute solver_dt and snapshot_stride
    # CFL: combine C++ h_min (or compute all-Python) with vp_max
    from preprocess.cfl_validator import compute_cfl_dt as _compute_cfl_dt, compute_solver_dt

    if accel_result["used_cpp"]:
        h_min_cpp = accel_result["cfl_dt"]  # C++ returns h_min, not full dt
        vp_max = float(vp_gll.max())
        cfl_dt = config.cfl_safety * h_min_cpp / vp_max
        logger.info(f"Using C++ h_min={h_min_cpp:.6e}, vp_max={vp_max:.6e}, cfl_dt={cfl_dt:.6e}")
    else:
        logger.info("Running CFL validation (Python)...")
        cfl_dt = _compute_cfl_dt(coords, vp_gll, config.cfl_safety)

    # Derive solver timestep and snapshot stride from output_dt_s
    solver_dt, snapshot_stride = compute_solver_dt(config.output_dt_s, cfl_dt)
    logger.info(
        f"  cfl_dt={cfl_dt:.6e}, solver_dt={solver_dt:.6e}, snapshot_stride={snapshot_stride}"
    )

    # Derive nsteps from total_duration_s
    nsteps = math.ceil(config.total_duration_s / solver_dt)
    total_duration_actual = nsteps * solver_dt
    if abs(total_duration_actual - config.total_duration_s) > 1e-12:
        logger.info(
            f"  Adjusted total_duration_s from {config.total_duration_s} to {total_duration_actual}"
        )

    # Step: source location
    logger.info("Locating source on free surface...")
    from preprocess.source_locator import locate_source

    source_z = float(domain_bounds["zmin"])
    source_x_m = float(config.source_x_m)
    source_y_m = float(config.source_y_m)
    source_xyz_arr = np.array([source_x_m, source_y_m, source_z], dtype=np.float64)
    src_result = locate_source(topology, source_xyz_arr, coords, boundary_tag, N)
    logger.info(
        f"  Source at ({source_x_m}, {source_y_m}, {source_z}), in {src_result['n_src_elem']} element(s)"
    )

    # Step: STF evaluation (uses solver_dt, not output_dt_s)
    logger.info("Evaluating STF...")
    try:
        from preprocess.stf_evaluator import evaluate_stf

        stf_t, stf_values = evaluate_stf(config.stf_func, solver_dt, nsteps)
    except ImportError:
        stf_t = np.arange(nsteps) * solver_dt
        stf_values = np.array([config.stf_func(t) for t in stf_t])

    # Step: pre-flight validation
    logger.info("Running pre-flight validation...")
    from preprocess.preflight import PreflightError, run_preflight

    try:
        strict = getattr(config, "strict_validation", True)
        preflight_result = run_preflight(
            topology,
            coords,
            jacobian,
            vp_gll,
            vs_gll,
            density_gll,
            boundary_tag,
            domain_bounds,
            config,
            source_xyz_arr,
            stf_values,
            cfl_dt,
            nsteps,
            snapshot_stride,
            n_gll,
            strict=strict,
        )
        logger.info(f"  {preflight_result.report()}")
    except PreflightError as e:
        logger.error(f"PREFLIGHT FAILED:\n{e}")
        sys.exit(1)

    # Step: PML damping
    if damping_cpp is not None:
        logger.info("Using C++-computed PML damping ramps")
        # Mask damping: zero out non-PML elements (C++ computes ramp for all)
        is_pml_flat = is_pml.reshape(-1)
        damping = damping_cpp.copy()
        for e in range(n_cell):
            if not is_pml_flat[e]:
                damping[e] = 0.0
    else:
        logger.info("Computing PML damping profiles...")
        try:
            from preprocess.pml import compute_pml_damping

            damping = compute_pml_damping(
                topology, coords, config.pml_thickness, domain_bounds, is_pml
            )
        except ImportError:
            damping = np.zeros((n_cell, n_gll, n_gll, n_gll), dtype=np.float64)
            logger.info("  pml.py not available — damping = 0")

    # Step: partition
    n_ranks = int(config.n_ranks)
    logger.info(f"Partitioning into {n_ranks} ranks...")
    try:
        from preprocess.partition import partition

        partition_result = partition(topology, coords, n_ranks)
    except ImportError:
        partition_result = None
        logger.info("  partition.py not available — skipping")

    # Step: build recording map
    logger.info("Building recording map...")
    try:
        from preprocess.recording_map import build_recording_map

        rd_max = float(config.record_depth_max_m)
        element_to_rank = partition_result.get("element_to_rank") if partition_result else None
        per_rank = partition_result.get("per_rank") if partition_result else None
        rec_map = build_recording_map(
            topology,
            domain_bounds,
            is_pml,
            rd_max,
            element_to_rank=element_to_rank,
            per_rank=per_rank,
        )
        logger.info(f"  record_depth_actual_m={rec_map['record_depth_actual_m']}")
    except ImportError:
        rec_map = None
        logger.info("  recording_map.py not available — skipping")

    # Step: write outputs
    # Precompute elastic coefficients from material properties (constant over time)
    mu_gll = density_gll * vs_gll ** 2
    lambda_gll = density_gll * (vp_gll ** 2 - 2.0 * vs_gll ** 2)

    fields = {
        "coords": coords,
        "jacobian": jacobian,
        "dxi_dx": dxi_dx,
        "mass": mass_gll,
        "vp": vp_gll,
        "vs": vs_gll,
        "density": density_gll,
        "lambda": lambda_gll,
        "mu": mu_gll,
        "is_pml": is_pml,
        "damping": damping,
    }

    logger.info(f"Writing model to: {mesh_path}")
    t0 = time.time()
    from preprocess.model_writer import write_model

    # Build tile config for model writer (tile_index in partition files + mesh.h5)
    tile_config = {
        "nx_elements": int(config.nx_elements),
        "ny_elements": int(config.ny_elements),
        "pml_xmin": int(config.pml_thickness.get("xmin", 0)),
        "pml_xmax": int(config.pml_thickness.get("xmax", 0)),
        "pml_ymin": int(config.pml_thickness.get("ymin", 0)),
        "pml_ymax": int(config.pml_thickness.get("ymax", 0)),
        "tilex_elements": list(config.tilex_elements),
        "tiley_elements": list(config.tiley_elements),
    }

    write_model(
        mesh_path,
        topology,
        fields,
        boundary_tag,
        domain_bounds,
        partition_result,
        recording_map=rec_map,
        tile_config=tile_config,
    )
    logger.debug(f"  model write: {time.time() - t0:.2f}s")

    config_h5 = os.path.join(os.path.dirname(mesh_path), "config.h5")
    logger.info(f"Writing config to: {config_h5}")
    t0 = time.time()
    from preprocess.config_writer import write_config

    write_config(
        config_h5,
        config,
        domain_bounds,
        stf_t,
        stf_values,
        source_xyz_arr,
        source_loc_result=src_result,
        solver_dt=solver_dt,
        snapshot_stride=snapshot_stride,
        nsteps=nsteps,
        recording_map=rec_map,
    )

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
