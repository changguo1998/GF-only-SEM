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


_PREPROCESS_BINARY: str | None = None
_PREPROCESS_BINARY: str | None = None


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
    global _PREPROCESS_BINARY, _PREPROCESS_BINARY
    _PREPROCESS_BINARY = _find_binary("gf_preprocess")
    _PREPROCESS_BINARY = _find_binary("gf_preprocess")


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
    if _PREPROCESS_BINARY is not None:
        # Ensure domain attrs for C++
        from preprocess.accelerator import _ensure_domain_attrs

        _ensure_domain_attrs(model_path, domain_bounds)

        pml = getattr(config, "pml_thickness", {}) or {}
        args = [
            os.path.abspath(model_path),
            "--N",
            str(N),
            "--cfl-safety",
            str(float(config.cfl_safety)),
            "--nx",
            str(int(getattr(config, "nx_elements", 0))),
            "--ny",
            str(int(getattr(config, "ny_elements", 0))),
        ]
        for face in ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]:
            thick = int(pml.get(face, 0))
            if thick > 0:
                args.extend([f"--pml-{face}", str(thick)])
        proc = _run_binary(_PREPROCESS_BINARY, ["stage1"] + args, desc="C++ stage1 (GLL+CFL)")
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

    if _PREPROCESS_BINARY is not None:
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


def _try_cpp_run(model_path, config, domain_bounds):
    """Try gf_preprocess run — unified C++ path covering the full pipeline."""
    if _PREPROCESS_BINARY is None:
        return False
    import h5py as _h5

    pml = getattr(config, "pml_thickness", {}) or {}
    args = [
        os.path.abspath(model_path),
        "--N",
        str(int(config.polynomial_order)),
        "--cfl-safety",
        str(float(config.cfl_safety)),
        "--nx",
        str(int(getattr(config, "nx_elements", 0))),
        "--ny",
        str(int(getattr(config, "ny_elements", 0))),
        "--n-ranks",
        str(int(getattr(config, "n_ranks", 1))),
    ]
    for face in ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]:
        thick = int(pml.get(face, 0))
        if thick > 0:
            args.extend([f"--pml-{face}", str(thick)])
    logger.info("Trying C++ unified run: %s run %s ...", _PREPROCESS_BINARY, model_path)
    proc = _run_binary(_PREPROCESS_BINARY, ["run"] + args, desc="C++ run (stage1→CPML→STF)")
    if proc is None:
        logger.info("C++ run failed — falling back to Python steps")
        return False
    return True


def write_attenuation_if_configured(model_path, config, n_cell, n_gll, logger=None) -> bool:
    """Step 10b (config-driven): SLS attenuation auto-injection.

    When ``config`` exposes ``q_mu``/``q_kappa`` (the canonical visco parameter
    settings), compute tau_sigma/tau_epsilon and write them to model.h5 so the
    forward solver auto-detects attenuation
    (forward/share/src/io.cpp: has_attenuation = !tau_sigma.empty()).
    Q -> infinity (elastic limit) => tau_epsilon == tau_sigma => solver output is
    bit-identical to elastic. ``n_sls`` is clamped to the solver-fixed N_SLS=3.

    Returns True when attenuation was written, False when the config carries no
    q_mu/q_kappa (skipped). ``logger`` is optional (None => silent).
    """
    q_mu_val = getattr(config, "q_mu", None)
    q_kappa_val = getattr(config, "q_kappa", None)
    if q_mu_val is None or q_kappa_val is None:
        if logger:
            logger.info("  SLS attenuation skipped (q_mu/q_kappa not set in config)")
        return False

    n_sls = int(getattr(config, "n_sls", 3))
    if n_sls != 3:
        if logger:
            logger.warning(
                "  n_sls=%d != solver fixed N_SLS=3 "
                "(forward/share/include/gf/attenuation.hpp) - clamping to 3",
                n_sls,
            )
        n_sls = 3
    f0_attenuation = float(getattr(config, "f0_for_pml_hz", 2.0))
    q_mu_f = float(q_mu_val)
    q_kappa_f = float(q_kappa_val)
    if logger:
        logger.info(
            f"Writing SLS attenuation (q_mu={q_mu_f:.3g}, q_kappa={q_kappa_f:.3g}, "
            f"n_sls={n_sls}, f0={f0_attenuation} Hz)..."
        )

    import numpy as np

    from preprocess.attenuation import write_attenuation_to_model

    q_mu_arr = np.full((n_cell, n_gll, n_gll, n_gll), q_mu_f, dtype=np.float64)
    q_kappa_arr = np.full((n_cell, n_gll, n_gll, n_gll), q_kappa_f, dtype=np.float64)
    write_attenuation_to_model(model_path, q_kappa_arr, q_mu_arr, n_gll, n_sls, f0_attenuation)
    if logger:
        logger.info("  attenuation write done")
    return True


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
    if _PREPROCESS_BINARY:
        logger.info(f"C++ preprocessor found: {_PREPROCESS_BINARY}")

    # ── Unified C++ path (gf_preprocess run) or per-step Python ──
    cpp_done = _try_cpp_run(model_path, config, domain_bounds)

    if cpp_done:
        logger.info("C++ run succeeded — reading results from HDF5")
        with h5py.File(model_path, "r") as f:
            fld = f["field/element"]
            coords = np.array(fld["coords"], dtype=np.float64)
            jacobian = np.array(fld["jacobian"], dtype=np.float64)
            dxi_dx = np.array(fld["dxi_dx"], dtype=np.float64)
            mass = np.array(fld["mass"], dtype=np.float64)
            vp = np.array(fld["vp"], dtype=np.float64)
            vs = np.array(fld["vs"], dtype=np.float64)
            density = np.array(fld["density"], dtype=np.float64)
            lam = np.array(fld["lambda"], dtype=np.float64)
            mu = np.array(fld["mu"], dtype=np.float64)
            is_pml = np.array(fld["is_pml"], dtype=np.bool_)
            damping = np.array(fld["damping"], dtype=np.float64)
            # C-PML
            cpml_K = np.array(fld["cpml_K"], dtype=np.float64)
            cpml_d = np.array(fld["cpml_d"], dtype=np.float64)
            cpml_alpha = np.array(fld["cpml_alpha"], dtype=np.float64)
            cpml_params = {"K": cpml_K, "d": cpml_d, "alpha": cpml_alpha}
            # STF
            if "config/stf_t" in f:
                stf_t = np.array(f["config/stf_t"], dtype=np.float64)
                stf_values = np.array(f["config/stf_values"], dtype=np.float64)
            else:
                stf_t = stf_values = np.array([], dtype=np.float64)
        h_min = None
        boundary_tag = step_boundary_detection(
            model_path,
            topology,
            config,
            domain_bounds,
            {
                "used_cpp": True,
                "coords": coords,
                "jacobian": jacobian,
                "dxi_dx": dxi_dx,
                "mass": mass,
                "h_min": h_min,
            },
        )
        gll = {
            "coords": coords,
            "jacobian": jacobian,
            "dxi_dx": dxi_dx,
            "mass": mass,
            "h_min": h_min,
            "used_cpp": True,
        }
        # Read solver_dt from C++ output (/info group in model.h5)
        solver_dt = 0.0
        snapshot_stride = 1
        nsteps = 0
        cfl_dt = 0.0
        with h5py.File(model_path, "r") as _f:
            if "info" in _f:
                info = _f["info"]
                if "solver_dt" in info.attrs:
                    solver_dt = float(info.attrs["solver_dt"])
                if "snapshot_stride" in info.attrs:
                    snapshot_stride = int(info.attrs["snapshot_stride"])
                if "nsteps" in info.attrs:
                    nsteps = int(info.attrs["nsteps"])
        if solver_dt <= 0.0:
            solver_dt = float(config.output_dt_s)
        if nsteps <= 0:
            nsteps = len(stf_t) if len(stf_t) > 0 else 0
        # source info from config.py
        source_z = getattr(config, "source_z_m", None)
        if source_z is None:
            source_z = float(domain_bounds["zmin"])
        source_xyz_arr = np.array(
            [config.source_x_m, config.source_y_m, source_z], dtype=np.float64
        )
        # ── Source location (Python: C++ didn't write cells to HDF5) ──
        try:
            from preprocess.source_locator import locate_source

            # Use already-loaded topology from main()
            source_xyz_arr = np.array(
                [float(config.source_x_m), float(config.source_y_m), float(source_z)],
                dtype=np.float64,
            )
            with h5py.File(model_path, "r") as _f:
                gll_coords_arr = np.array(_f["field/element/coords"], dtype=np.float64)
                btag = np.array(_f["field/surface/boundary_tag"], dtype=np.int64)
                if "field/element/is_pml" in _f:
                    is_pml_arr = np.array(_f["field/element/is_pml"], dtype=np.bool_)
                else:
                    is_pml_arr = None
            src_result = locate_source(
                topology, source_xyz_arr, gll_coords_arr, btag, N, is_pml_arr
            )
            logger.info(f"  Source in {src_result['n_src_cell']} element(s)")
        except (ImportError, Exception) as _e:
            logger.warning(f"  Source location failed: {_e}")
            import traceback

            traceback.print_exc()
            src_result = {
                "n_src_cell": 0,
                "cell_ids": np.array([], dtype=np.int64),
                "xi": np.array([], dtype=np.float64),
                "eta": np.array([], dtype=np.float64),
                "zeta": np.array([], dtype=np.float64),
                "weights": np.array([], dtype=np.float64),
                "mode": "surface",
            }

        # ── Step 8: Partition (read C++ results, compute per_rank) ──
        n_ranks = int(config.n_ranks)
        logger.info(f"Building partition data from C++ results ({n_ranks} ranks)...")
        try:
            import h5py as _h5

            with _h5.File(model_path, "r") as pf:
                if "partition/element_to_rank" in pf:
                    element_to_rank_arr = np.array(pf["partition/element_to_rank"], dtype=np.int64)
                else:
                    element_to_rank_arr = np.zeros(n_cell, dtype=np.int64)
                if "partition/global_cell2global_node" in pf:
                    gcn4d = np.array(pf["partition/global_cell2global_node"], dtype=np.int32)
                    n_global_node = int(gcn4d.max()) + 1 if gcn4d.size > 0 else 0
                else:
                    gcn4d = np.zeros((n_cell, n_gll, n_gll, n_gll), dtype=np.int32)
                    n_global_node = 0

            from preprocess.partition import compute_per_rank

            per_rank = compute_per_rank(topology, n_gll, element_to_rank_arr, gcn4d)
            partition_result = {
                "element_to_rank": element_to_rank_arr,
                "n_ranks": n_ranks,
                "per_rank": per_rank,
                "global_cell2global_node": gcn4d,
                "n_global_node": n_global_node,
            }
            logger.info(f"  {n_ranks} ranks, {n_global_node} global nodes")
        except ImportError:
            partition_result = None
            logger.info("  partition.py not available — skipping")
    else:
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

        # ── Step 4: Material interpolation ──
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

        # ── Step 5b: C-PML parameters ──
        f0_for_pml = getattr(config, "f0_for_pml_hz", 2.0)
        pml_thickness_cfg = getattr(config, "pml_thickness", {}) or {}
        nx_el = int(getattr(config, "nx_elements", 1))
        ny_el = int(getattr(config, "ny_elements", 1))
        nz_el = n_cell // (nx_el * ny_el) if nx_el * ny_el > 0 else 1
        dx_el = (domain_bounds["xmax"] - domain_bounds["xmin"]) / max(nx_el, 1)
        dy_el = (domain_bounds["ymax"] - domain_bounds["ymin"]) / max(ny_el, 1)
        dz_el = (domain_bounds["zmax"] - domain_bounds["zmin"]) / max(nz_el, 1)
        pml_widths = {
            "xmin": pml_thickness_cfg.get("xmin", 0) * dx_el,
            "xmax": pml_thickness_cfg.get("xmax", 0) * dx_el,
            "ymin": pml_thickness_cfg.get("ymin", 0) * dy_el,
            "ymax": pml_thickness_cfg.get("ymax", 0) * dy_el,
            "zmin": pml_thickness_cfg.get("zmin", 0) * dz_el,
            "zmax": pml_thickness_cfg.get("zmax", 0) * dz_el,
        }
        logger.info(f"Computing C-PML parameters (f0={f0_for_pml} Hz, dt={solver_dt:.4e} s)...")
        from preprocess.pml_cpml import compute_cpml_parameters

        cpml_params = compute_cpml_parameters(
            coords, is_pml, domain_bounds, pml_widths, vp, f0_for_pml, solver_dt
        )
        n_pml_val = int(is_pml.sum())
        logger.info(
            f"  PML elements: {n_pml_val}, regions: {np.unique(cpml_params['pml_region'])}"
        )

        # ── Step 6: Source location ──
        source_z = getattr(config, "source_z_m", None)
        if source_z is None:
            source_z = float(domain_bounds["zmin"])
            logger.info("Locating source on free surface...")
        else:
            logger.info(f"Locating BURIED source at depth z={source_z} m...")
        from preprocess.source_locator import locate_source

        source_xyz_arr = np.array(
            [config.source_x_m, config.source_y_m, source_z], dtype=np.float64
        )
        src_result = locate_source(
            topology, source_xyz_arr, coords, boundary_tag, N, is_pml=is_pml
        )
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

            partition_result = partition(topology, n_gll, n_ranks)
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
        global_cell2global_node = (
            partition_result.get("global_cell2global_node") if partition_result else None
        )
        rec_map = build_recording_map(
            topology,
            domain_bounds,
            is_pml,
            rd_max,
            global_cell2global_node=global_cell2global_node,
            gll_node_coords=coords,
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
    fields.update(cpml_params)
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

    # ── Step 10b: SLS attenuation (config-driven, visco parameter settings) ──
    # When config exposes q_mu/q_kappa, compute tau_sigma/tau_epsilon and write them
    # to model.h5 so the solver auto-detects attenuation (n_sls clamped to 3).
    write_attenuation_if_configured(model_path, config, n_cell, n_gll, logger)

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
