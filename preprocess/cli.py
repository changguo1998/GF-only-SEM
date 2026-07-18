"""Preprocessor entry point — adaptive Python + C++ accelerator pipeline.

Linear workflow: each step function checks if C++ accelerator is available.
If available: writes inputs to HDF5, runs C++ binary, reads outputs from HDF5.
If unavailable: runs pure Python implementation.
"""

from __future__ import annotations

import logging
import math
import os
import shlex
import subprocess
import sys
import time

import h5py
import numpy as np

from preprocess.config_loader import load_config
from preprocess.topology_reader import read_topology

# ── Logging ──


def setup_logging(log_dir: str = "log") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "preprocess.log")
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[preprocess] %(message)s"))
    logger.addHandler(ch)
    logger.info(f"Log file: {os.path.abspath(log_path)}")
    return logger


logger: logging.Logger | None = None


# ── Accelerator binary discovery ──


_STAGE1_BINARY: str | None = None
_STAGE2_BINARY: str | None = None


def _find_binary(name: str, extra_dirs: list[str] | None = None) -> str | None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir)
    candidates: list[str] = []
    # Project bin/
    candidates.append(os.path.join(project_root, "bin", name))
    # Source-adjacent
    candidates.append(os.path.join(this_dir, "cpp", name))
    candidates.append(os.path.join(this_dir, "cpp", "bin", name))
    candidates.append(os.path.join(this_dir, "cpp", "build", name))
    # PATH
    candidates.append(name)
    # Extra
    if extra_dirs:
        for d in extra_dirs:
            candidates.append(os.path.join(d, name))
    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.abspath(cand)
    return None


def _init_accelerators() -> None:
    global _STAGE1_BINARY, _STAGE2_BINARY
    _STAGE1_BINARY = _find_binary("gf_preprocess_cpp")
    _STAGE2_BINARY = _find_binary("gf_preprocess_stage2")


def _run_binary(
    binary: str, args: list[str], timeout: int = 600, desc: str = ""
) -> subprocess.CompletedProcess | None:
    cmd = [binary] + args
    if desc:
        logger.info(f"Running {desc}: {' '.join(shlex.quote(str(x)) for x in cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        logger.warning(f"Binary not found: {binary}")
        return None
    except subprocess.TimeoutExpired:
        logger.warning(f"{desc or binary} timed out after {timeout}s")
        return None
    if proc.returncode != 0:
        logger.warning(f"{desc or binary} exit code {proc.returncode}")
        if proc.stderr.strip():
            for line in proc.stderr.strip().split("\n")[-5:]:
                logger.warning(f"  stderr: {line}")
        return None
    if proc.stderr.strip():
        for line in proc.stderr.strip().split("\n"):
            logger.debug(f"  [{desc or 'C++'}] {line}")
    return proc


# ── Adaptive step functions ──


def step_gll_geometry(
    model_path: str, topology: object, config: object, domain_bounds: dict[str, float]
) -> dict:
    N = int(config.polynomial_order)
    """Compute GLL geometry + CFL h_min. C++ accelerator if available."""
    if _STAGE1_BINARY is not None:
        # Ensure domain attrs for C++
        from preprocess.accelerator import _ensure_domain_attrs

        _ensure_domain_attrs(model_path, domain_bounds)

        pml = getattr(config, "pml_thickness", {}) or {}
        args = [
            os.path.abspath(model_path),
            str(N),
            str(float(config.cfl_safety)),
            str(int(getattr(config, "nx_elements", 0))),
            str(int(getattr(config, "ny_elements", 0))),
            str(int(pml.get("xmin", 0))),
            str(int(pml.get("xmax", 0))),
            str(int(pml.get("ymin", 0))),
            str(int(pml.get("ymax", 0))),
            str(int(pml.get("zmin", 0))),
            str(int(pml.get("zmax", 0))),
        ]
        proc = _run_binary(_STAGE1_BINARY, args, desc="C++ stage1 (GLL+CFL)")
        if proc is not None:
            # Parse CFL info from stdout
            h_min = None
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("H_MIN="):
                    h_min = float(line.split("=", 1)[1])
                elif line.startswith("CFL_DT="):
                    cfl_dt_cpp = float(line.split("=", 1)[1])
            if h_min is None:
                logger.warning("C++ stage1 didn't print H_MIN — falling to Python")
            else:
                # Read results from HDF5
                with h5py.File(model_path, "r") as f:
                    fld = f["field/element"]
                    coords = np.array(fld["coords"], dtype=np.float64)
                    jacobian = np.array(fld["jacobian"], dtype=np.float64)
                    dxi_dx = np.array(fld["dxi_dx"], dtype=np.float64)
                    mass = np.array(fld["mass"], dtype=np.float64)
                logger.info(f"  C++ GLL: coords={coords.shape}, h_min={h_min:.4e}")
                return {
                    "coords": coords,
                    "jacobian": jacobian,
                    "dxi_dx": dxi_dx,
                    "mass": mass,
                    "h_min": h_min,
                    "used_cpp": True,
                }

    # Python fallback
    logger.info("Computing GLL geometry (Python)...")
    from preprocess.gll_geometry import compute_gll_geometry

    t0 = time.time()
    coords, jacobian, dxi_dx, mass = compute_gll_geometry(topology, N)
    # Compute h_min via compute_cfl_dt with unit vp (isolates h_min)
    from preprocess.cfl_validator import compute_cfl_dt

    unit_vp = np.ones(coords.shape[:-1], dtype=np.float64)
    h_min = compute_cfl_dt(coords, unit_vp, 1.0)  # cfl_safety=1, vp=1 => cfl_dt = h_min
    logger.info(f"  Python GLL: {time.time() - t0:.2f}s, h_min={h_min:.4e}")
    return {
        "coords": coords,
        "jacobian": jacobian,
        "dxi_dx": dxi_dx,
        "mass": mass,
        "h_min": h_min,
        "used_cpp": False,
    }


def step_boundary_detection(
    model_path: str,
    topology: object,
    config: object,
    domain_bounds: dict[str, float],
    gll_result: dict,
) -> np.ndarray:
    """Detect free/absorbing boundaries. C++ accelerator result if available."""
    # If C++ stage1 ran, boundary_tag is already in HDF5
    if gll_result.get("used_cpp"):
        with h5py.File(model_path, "r") as f:
            if "field/surface/boundary_tag" in f:
                bt = np.array(f["field/surface/boundary_tag"], dtype=np.int64)
                logger.info(f"Using C++ boundary_tag ({bt.shape[0]} surfaces)")
                return bt

    # Python fallback
    logger.info("Detecting boundaries (Python)...")
    from preprocess.boundary_detector import detect_boundaries

    boundary_tag, _ = detect_boundaries(topology, domain_bounds)
    return boundary_tag


def step_pml(
    model_path: str,
    topology: object,
    config: object,
    domain_bounds: dict[str, float],
    coords: np.ndarray,
    gll_result: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """PML expansion + damping. C++ accelerator result if available."""
    n_cell = topology.n_cell
    N = int(config.polynomial_order)
    n_gll = N + 1

    # If C++ stage1 ran, is_pml and damping are in HDF5
    if gll_result.get("used_cpp"):
        with h5py.File(model_path, "r") as f:
            fld = f.get("field/element")
            if fld is not None and "is_pml" in fld and "damping" in fld:
                is_pml = np.array(fld["is_pml"], dtype=np.bool_)
                damping = np.array(fld["damping"], dtype=np.float64)
                logger.info(f"Using C++ PML: is_pml count={int(is_pml.sum())}")
                return is_pml, damping

    # Python fallback: PML expansion
    logger.info("Expanding PML (Python)...")
    pml_thickness_cfg = getattr(config, "pml_thickness", {}) or {}
    nx = int(getattr(config, "nx_elements", 0))
    ny = int(getattr(config, "ny_elements", 0))
    nz = n_cell // (nx * ny) if nx * ny > 0 else 0

    if nx * ny * nz == n_cell:
        is_pml = np.zeros(n_cell, dtype=np.bool_)
        cnt = {
            "xmin": int(pml_thickness_cfg.get("xmin", 0)),
            "xmax": int(pml_thickness_cfg.get("xmax", 0)),
            "ymin": int(pml_thickness_cfg.get("ymin", 0)),
            "ymax": int(pml_thickness_cfg.get("ymax", 0)),
            "zmax": int(pml_thickness_cfg.get("zmax", 0)),
        }
        for e in range(n_cell):
            i = e % nx
            j = (e // nx) % ny
            k = e // (nx * ny)
            if (
                i < cnt["xmin"]
                or i >= nx - cnt["xmax"]
                or j < cnt["ymin"]
                or j >= ny - cnt["ymax"]
                or k >= nz - cnt["zmax"]
            ):
                is_pml[e] = True
    else:
        from preprocess.boundary_detector import detect_boundaries

        _, is_pml = detect_boundaries(topology, domain_bounds)

    # PML damping (Python)
    logger.info("Computing PML damping (Python)...")
    try:
        from preprocess.pml import compute_pml_damping

        damping = compute_pml_damping(topology, coords, pml_thickness_cfg, domain_bounds, is_pml)
    except ImportError:
        damping = np.zeros((n_cell, n_gll, n_gll, n_gll), dtype=np.float64)
        logger.info("  pml.py not available — damping = 0")

    return is_pml, damping


def step_material_interpolation(config: object, coords: np.ndarray) -> tuple:
    """Interpolate material properties (always Python — user callables)."""
    material_model_path = getattr(config, "model_path", None)
    logger.info("Loading material model (Python)...")
    t0 = time.time()
    from preprocess.model_loader import load_and_interpolate

    vp, vs, density = load_and_interpolate(material_model_path, coords, config=config)
    logger.info(f"  material interpolation: {time.time() - t0:.2f}s")
    return vp, vs, density


def step_lame_and_cfl(
    model_path: str,
    config: object,
    vp: np.ndarray,
    vs: np.ndarray,
    density: np.ndarray,
    coords: np.ndarray,
    h_min: float,
) -> dict:
    N = int(config.polynomial_order)
    """Compute λ/μ, CFL solver_dt, nsteps. C++ stage2 if available."""

    if _STAGE2_BINARY is not None:
        # Write vp/vs/density + config to HDF5, run stage2
        logger.info("Writing material arrays for C++ stage2...")
        with h5py.File(model_path, "a") as f:
            fld = f.require_group("field/element")
            for name, arr in [("vp", vp), ("vs", vs), ("density", density)]:
                if name in fld:
                    del fld[name]
                fld.create_dataset(name, data=arr, compression=None)
            cfg = f.require_group("config")
            cfg_attrs = {
                "cfl_safety": float(config.cfl_safety),
                "output_dt_s": float(config.output_dt_s),
                "total_duration_s": float(config.total_duration_s),
                "n_ranks": int(config.n_ranks),
                "snapshot_precision": np.bytes_(config.snapshot_precision.encode()),
                "storage_limit_gb": float(getattr(config, "storage_limit_gb", 100.0)),
                "record_depth_max_m": float(config.record_depth_max_m),
                "nx_elements": int(config.nx_elements),
                "ny_elements": int(config.ny_elements),
                "NGLL": N + 1,
            }
            for key, val in cfg_attrs.items():
                cfg.attrs[key] = val

        from preprocess.stage2_runner import run_stage2

        try:
            stats = run_stage2(model_path)
        except RuntimeError as e:
            logger.warning(f"C++ stage2 failed: {e}")
            stats = None

        if stats:
            # Read results from HDF5
            with h5py.File(model_path, "r") as f:
                lam = np.array(f["field/element/lambda"])
                mu = np.array(f["field/element/mu"])
            solver_dt = float(stats.get("STAT_SOLVER_DT", "0"))
            snapshot_stride = int(stats.get("STAT_SNAPSHOT_STRIDE", "1"))
            nsteps = int(stats.get("STAT_NSTEPS", "0"))
            cfl_dt = float(stats.get("STAT_CFL_DT", "0"))

            # Delete temp material arrays
            with h5py.File(model_path, "a") as f:
                fld = f["field/element"]
                for name in ["vp", "vs", "density"]:
                    if name in fld:
                        del fld[name]

            logger.info(f"  C++ λ/μ: solver_dt={solver_dt:.6e}, nsteps={nsteps}")
            logger.info(f"  C++ stats: λ min={stats.get('STAT_LAM_MIN', '?')}")
            return {
                "lam": lam,
                "mu": mu,
                "solver_dt": solver_dt,
                "snapshot_stride": snapshot_stride,
                "nsteps": nsteps,
                "cfl_dt": cfl_dt,
                "used_cpp": True,
            }

    # Python fallback
    logger.info("Computing λ/μ (Python)...")
    mu = density * vs**2
    lam = density * (vp**2 - 2.0 * vs**2)

    logger.info("Computing CFL (Python)...")
    from preprocess.cfl_validator import compute_solver_dt

    vp_max = float(vp.max())
    cfl_dt = float(config.cfl_safety) * h_min / vp_max
    solver_dt, snapshot_stride = compute_solver_dt(float(config.output_dt_s), cfl_dt)
    nsteps = math.ceil(float(config.total_duration_s) / solver_dt)
    logger.info(f"  cfl_dt={cfl_dt:.6e}, solver_dt={solver_dt:.6e}, nsteps={nsteps}")

    return {
        "lam": lam,
        "mu": mu,
        "solver_dt": solver_dt,
        "snapshot_stride": snapshot_stride,
        "nsteps": nsteps,
        "cfl_dt": cfl_dt,
        "used_cpp": False,
    }


# ── Main pipeline ──


def main() -> None:
    global logger
    logger = setup_logging()
    start = time.time()

    model_path = os.path.abspath("model.h5")
    config_path = os.path.abspath("config.py")

    logger.info(f"Loading config: {config_path}")
    config = load_config(config_path)

    logger.info(f"Reading topology from: {model_path}")
    topology = read_topology(model_path)
    n_cell = topology.n_cell
    N = int(config.polynomial_order)
    n_gll = N + 1

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

    # Init accelerators
    _init_accelerators()
    if _STAGE1_BINARY:
        logger.info(f"C++ stage1 found: {_STAGE1_BINARY}")
    if _STAGE2_BINARY:
        logger.info(f"C++ stage2 found: {_STAGE2_BINARY}")

    # ── Step 1: GLL geometry + CFL h_min ──
    gll = step_gll_geometry(model_path, topology, config, domain_bounds)
    coords = gll["coords"]
    jacobian = gll["jacobian"]
    dxi_dx = gll["dxi_dx"]
    mass = gll["mass"]
    h_min = gll["h_min"]

    # ── Step 2: Boundary detection ──
    boundary_tag = step_boundary_detection(model_path, topology, config, domain_bounds, gll)

    # ── Step 3: PML ──
    is_pml, damping = step_pml(model_path, topology, config, domain_bounds, coords, gll)

    # ── Step 4: Material interpolation (always Python) ──
    vp, vs, density = step_material_interpolation(config, coords)
    mass = mass * density

    # ── Step 5: λ/μ + CFL solver_dt ──
    lame = step_lame_and_cfl(model_path, config, vp, vs, density, coords, h_min)
    lam = lame["lam"]
    mu = lame["mu"]
    solver_dt = lame["solver_dt"]
    snapshot_stride = lame["snapshot_stride"]
    nsteps = lame["nsteps"]
    cfl_dt = lame["cfl_dt"]

    # ── Step 6: Source location ──
    source_z = getattr(config, "source_z_m", None)
    if source_z is None:
        source_z = float(domain_bounds["zmin"])
        logger.info("Locating source on free surface...")
    else:
        logger.info(f"Locating BURIED source at depth z={source_z} m...")
    from preprocess.source_locator import locate_source

    source_xyz_arr = np.array([config.source_x_m, config.source_y_m, source_z], dtype=np.float64)
    # is_pml is needed for buried mode to exclude PML elements
    src_result = locate_source(topology, source_xyz_arr, coords, boundary_tag, N, is_pml=is_pml)
    mode_label = "BURIED" if source_z != float(domain_bounds["zmin"]) else "on free surface"
    logger.info(
        f"  Source at ({config.source_x_m}, {config.source_y_m}, {source_z}), "
        f"{mode_label}, in {src_result['n_src_cell']} element(s)"
    )

    # ── Step 7: STF ──
    logger.info("Evaluating STF...")
    try:
        from preprocess.stf_evaluator import evaluate_stf

        stf_t, stf_values = evaluate_stf(config.stf_func, solver_dt, nsteps)
    except ImportError:
        stf_t = np.arange(nsteps) * solver_dt
        stf_values = np.array([config.stf_func(t) for t in stf_t])

    # ── Step 8: Partition ──
    n_ranks = int(config.n_ranks)
    logger.info(f"Partitioning into {n_ranks} ranks...")
    try:
        from preprocess.partition import partition

        partition_result = partition(topology, coords, n_ranks)
    except ImportError:
        partition_result = None
        logger.info("  partition.py not available — skipping")

    # ── Step 9: Recording map ──
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

    # ── Step 10: Write outputs ──
    fields = {
        "coords": coords,
        "jacobian": jacobian,
        "dxi_dx": dxi_dx,
        "mass": mass,
        "vp": vp,
        "vs": vs,
        "density": density,
        "lambda": lam,
        "mu": mu,
        "is_pml": is_pml,
        "damping": damping,
    }
    tile_config = {
        "nx_elements": int(config.nx_elements),
        "ny_elements": int(config.ny_elements),
        "pml_xmin": int(config.pml_thickness.get("xmin", 0)),
        "pml_xmax": int(config.pml_thickness.get("xmax", 0)),
        "pml_ymin": int(config.pml_thickness.get("ymin", 0)),
        "pml_ymax": int(config.pml_thickness.get("ymax", 0)),
        "tilex_elements": list(config.tilex_elements),
        "tiley_elements": list(config.tiley_elements),
        "domain_bounds": domain_bounds,
        "record_depth_actual_m": rec_map.get("record_depth_actual_m", 0.0) if rec_map else 0.0,
    }

    logger.info(f"Writing model to: {model_path}")
    t0 = time.time()
    from preprocess.model_writer import write_model

    write_model(
        model_path,
        topology,
        fields,
        boundary_tag,
        domain_bounds,
        partition_result,
        recording_map=rec_map,
        tile_config=tile_config,
    )
    logger.info(f"  model write: {time.time() - t0:.2f}s")

    config_h5 = os.path.join(os.path.dirname(model_path), "config.h5")
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
    logger.info(f"  config write: {time.time() - t0:.2f}s")

    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
