"""Accelerator — wraps gf_preprocess_cpp C++ executable for heavy computations.

Checks for the compiled binary, runs it as a subprocess, reads precomputed
GLL geometry + CFL data from HDF5, and passes results back to the Python
pipeline.  Falls back to pure Python if the binary is unavailable.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
import types
from collections.abc import Callable

import h5py
import numpy as np
import numpy.typing as npt

logger = logging.getLogger("preprocess")


def _find_binary() -> str | None:
    """Locate gf_preprocess_cpp binary.

    Search order:
      1. Same directory as this module (preprocess/cpp/...)
      2. PATH
      3. Build directory adjacent to source (preprocess/cpp/build/...)
    Returns absolute path or None.
    """
    # Same directory as this module
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(this_dir, "cpp", "gf_preprocess_cpp"),
        os.path.join(this_dir, "cpp", "build", "gf_preprocess_cpp"),
        "gf_preprocess_cpp",  # via PATH
    ]

    for cand in candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            logger.debug(f"Found C++ accelerator: {cand}")
            return os.path.abspath(cand)
        # Also try with .exe suffix (Windows/MinGW)
        cand_exe = cand + ".exe"
        if os.path.isfile(cand_exe) and os.access(cand_exe, os.X_OK):
            logger.debug(f"Found C++ accelerator: {cand_exe}")
            return os.path.abspath(cand_exe)

    # Check build directory relative to project root
    project_root = os.path.dirname(this_dir)  # preprocess/ is one level down
    build_candidates = [
        os.path.join(project_root, "build", "preprocess", "cpp", "gf_preprocess_cpp"),
        os.path.join(project_root, "build", "preprocess", "cpp", "gf_preprocess_cpp.exe"),
        os.path.join(project_root, "build", "preprocess", "cpp", "Debug", "gf_preprocess_cpp.exe"),
    ]
    for cand in build_candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            logger.debug(f"Found C++ accelerator: {cand}")
            return os.path.abspath(cand)

    logger.info("C++ accelerator not found — using pure Python")
    return None


def run_accelerator(
    mesh_path: str,
    config: types.ModuleType,
    domain_bounds: dict[str, float],
) -> dict[str, any]:
    """Run gf_preprocess_cpp if available, return precomputed data.

    Args:
        mesh_path: Path to mesh.h5 (with /topology/ group).
        config: Loaded config module.
        domain_bounds: Dict with xmin, xmax, ymin, ymax, zmin, zmax.

    Returns:
        Dict with keys:
            used_cpp: bool — whether C++ was used
            coords: [n_cell, NGLL, NGLL, NGLL, 3] float64 or None
            dxi_dx: [n_cell, NGLL, NGLL, NGLL, 9] float64 or None
            jacobian: [n_cell, NGLL, NGLL, NGLL] float64 or None
            mass: [n_cell, NGLL, NGLL, NGLL] float64 or None
            cfl_dt: float (CFL-limited timestep from C++, or None)
            h_min: float (minimum GLL spacing, or None)
            solver_dt: float (derived solver timestep, or None)
            snapshot_stride: int (or None)
    """
    result: dict[str, any] = {
        "used_cpp": False,
        "coords": None,
        "dxi_dx": None,
        "jacobian": None,
        "mass": None,
        "damping": None,
        "cfl_dt": None,
        "h_min": None,
        "solver_dt": None,
        "snapshot_stride": None,
    }

    binary = _find_binary()
    if binary is None:
        return result

    N = int(config.polynomial_order)
    cfl_safety = float(config.cfl_safety)

    # PML thickness — default to 0 if not present
    pml = getattr(config, "pml_thickness", None)
    if pml is None:
        pml = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}

    # Ensure domain attrs exist in mesh.h5 (C++ reads them from /domain/ attrs)
    _ensure_domain_attrs(mesh_path, domain_bounds)

    # Build command
    cmd = [
        binary,
        os.path.abspath(mesh_path),
        str(N),
        str(cfl_safety),
        str(int(pml.get("xmin", 0))),
        str(int(pml.get("xmax", 0))),
        str(int(pml.get("ymin", 0))),
        str(int(pml.get("ymax", 0))),
        str(int(pml.get("zmin", 0))),
        str(int(pml.get("zmax", 0))),
    ]

    logger.info(f"Running C++ accelerator: {' '.join(shlex.quote(str(x)) for x in cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )
    except FileNotFoundError:
        logger.warning("C++ accelerator binary not found after all — falling back to Python")
        return result
    except subprocess.TimeoutExpired:
        logger.warning("C++ accelerator timed out — falling back to Python")
        return result

    if proc.returncode != 0:
        logger.warning(
            f"C++ accelerator exited with code {proc.returncode}:\n"
            f"  stderr: {proc.stderr[:500]}"
        )
        return result

    # Parse stdout for CFL info
    h_min = None
    cfl_dt = None
    for line in proc.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("H_MIN="):
            h_min = float(line.split("=", 1)[1])
        elif line.startswith("CFL_DT="):
            cfl_dt = float(line.split("=", 1)[1])
        elif line.startswith("CFL_SAFETY="):
            cfl_safety_parsed = float(line.split("=", 1)[1])
            # verify consistency
            if abs(cfl_safety_parsed - cfl_safety) > 1e-15:
                logger.warning(
                    f"cfl_safety mismatch: config={cfl_safety}, C++={cfl_safety_parsed}"
                )

    # Log stderr for diagnostics
    if proc.stderr.strip():
        for line in proc.stderr.strip().split("\n"):
            logger.debug(f"  [C++] {line}")

    if h_min is None or cfl_dt is None:
        logger.warning("C++ accelerator did not print CFL info — falling back to Python")
        return result

    # Read precomputed fields from HDF5
    try:
        with h5py.File(mesh_path, "r") as f:
            fld = f.get("field/element")
            if fld is None:
                logger.warning("C++ accelerator didn't write field/element — falling back")
                return result

            n_cell = f["topology"].attrs["n_cell"]
            NGLL = N + 1

            coords = np.array(fld["coords"], dtype=np.float64)
            if coords.shape != (n_cell, NGLL, NGLL, NGLL, 3):
                logger.warning(
                    f"C++ coords shape mismatch: {coords.shape} != ({n_cell},{NGLL},{NGLL},{NGLL},3)"
                )
                return result

            dxi_dx = np.array(fld["dxi_dx"], dtype=np.float64)
            jacobian = np.array(fld["jacobian"], dtype=np.float64)
            mass = np.array(fld["mass"], dtype=np.float64)

            has_damping = "damping" in fld
            damping = np.array(fld["damping"], dtype=np.float64) if has_damping else None

    except Exception as e:
        logger.warning(f"Failed to read C++ results from HDF5: {e}")
        return result

    result.update(
        {
            "used_cpp": True,
            "coords": coords,
            "dxi_dx": dxi_dx,
            "jacobian": jacobian,
            "mass": mass,
            "damping": damping,
            "cfl_dt": cfl_dt,
            "h_min": h_min,
        }
    )

    logger.info(
        f"C++ accelerator: h_min={h_min:.6e}, cfl_dt={cfl_dt:.6e}, "
        f"coords={coords.shape}, dxi_dx={dxi_dx.shape}"
    )

    return result


def _ensure_domain_attrs(mesh_path: str, domain_bounds: dict[str, float]) -> None:
    """Ensure /domain/ attributes exist in mesh.h5 (C++ reads them)."""
    try:
        with h5py.File(mesh_path, "a") as f:
            dom = f.require_group("domain")
            for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
                if key not in dom.attrs:
                    dom.attrs[key] = float(domain_bounds[key])
    except Exception:
        logger.warning("Could not write domain attrs to mesh.h5", exc_info=True)