#!/usr/bin/env python3
"""Compare SEM Green's function tiles against analytical full-space Stokes solution.

For a full-space setup (all PML boundaries, source away from boundaries),
the full-space Stokes solution is the exact reference — no free-surface
reflections to exclude. Comparison uses the entire waveform.

For a half-space setup (free surface at z=0), comparison is time-windowed
to before the first surface-reflected phase.

PARAMETER CONSISTENCY (source of truth = the SEM run artifacts):

All analytical-side physical parameters are read from the SEM run's own
output files instead of being duplicated here:

  * source time function  -> config.h5:/source/stf_values + /source/stf_t
                             (the exact STF the solver used, sampled at
                             solver_dt by the preprocessor; the force
                             amplitude is embedded in these values)
  * source location       -> config.h5:/source/{x,y,z}
  * time step             -> config.h5:/simulation/output_dt_s
  * material (vp, vs, rho)-> model.h5:/field/cell/{vp,vs,density}
                             (representative value = median)

The comparison is therefore consistent with the SEM run BY CONSTRUCTION,
eliminating the class of hardcoded-constant drift bugs. A hard assertion
checks that the recorded tile time step equals output_dt_s before using
the config STF.

Usage:
    python3 analytical_compare.py <greenfun_dir> [--source x y z]
        [--config-h5 PATH] [--model-h5 PATH] [--fullspace]
"""

import glob
import os
import sys

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from analytical_green import (  # noqa: E402
    first_surface_reflected_arrival_time_s,
    relative_l2_error,
    stokes_displacement_green_tensor,
    wavelet_correlation,
)

FORCE_LABELS = ["x", "y", "z"]
MIN_ACCEPTABLE_SCALE = 0.8
MAX_ACCEPTABLE_SCALE = 1.2


def _to_float(value, name: str) -> float:
    """Convert a value to float, giving a descriptive error on invalid input.

    All numeric conversions in this script go through here so a corrupt or
    mismatched SEM artifact fails with a named verdict instead of a bare
    ValueError.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot parse {name} as float: {value!r}") from exc


def _to_int(value, name: str) -> int:
    """Convert a value to int, giving a descriptive error on invalid input."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot parse {name} as int: {value!r}") from exc


def find_artifacts(
    greenfun_dir: str, config_h5: str | None, model_h5: str | None
) -> tuple[str, str]:
    """Locate config.h5 / model.h5 (given path, else greenfun_dir's parent)."""
    case_dir = os.path.abspath(os.path.join(greenfun_dir, os.pardir))
    config_path = config_h5 or os.path.join(case_dir, "config.h5")
    model_path = model_h5 or os.path.join(case_dir, "model.h5")
    missing = [p for p in (config_path, model_path) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "SEM artifacts not found: "
            + ", ".join(missing)
            + " (pass --config-h5/--model-h5, or keep them next to the tiles)"
        )
    return config_path, model_path


def load_sem_parameters(config_h5: str, model_h5: str, n_steps: int) -> dict:
    """Read the SEM parameters the analytical solution must reproduce."""
    with h5py.File(config_h5, "r") as f:
        if not ("source/stf_t" in f and "source/stf_values" in f):
            raise KeyError(f"{config_h5}: missing /source/stf_t or /source/stf_values")
        stf_time = np.asarray(f["source/stf_t"], dtype=np.float64)
        stf_values = np.asarray(f["source/stf_values"], dtype=np.float64)
        source_xyz = np.array(
            [
                _to_float(np.asarray(f["source"].attrs["x"]), "source.x"),
                _to_float(np.asarray(f["source"].attrs["y"]), "source.y"),
                _to_float(np.asarray(f["source"].attrs["z"]), "source.z"),
            ],
            dtype=np.float64,
        )
        output_dt_s = _to_float(
            np.asarray(f["simulation"].attrs["output_dt_s"]), "simulation.output_dt_s"
        )
        nsteps_artifact = _to_int(np.asarray(f["simulation"].attrs["nsteps"]), "simulation.nsteps")

    with h5py.File(model_h5, "r") as f:

        def _median(path: str) -> float:
            data = np.asarray(f[path], dtype=np.float64)
            return _to_float(np.median(data.ravel()), "model material")

        vp = _median("field/cell/vp")
        vs = _median("field/cell/vs")
        rho = _median("field/cell/density")

    # Validate STF time grid against output_dt_s.
    stf_dt = (
        _to_float(stf_time[1] - stf_time[0], "stf_time.dt") if len(stf_time) > 1 else output_dt_s
    )
    if not np.isclose(stf_dt, output_dt_s, rtol=1e-9, atol=1e-12):
        raise ValueError(
            f"config.h5 STF dt={stf_dt} != output_dt_s={output_dt_s} — STF/record misalignment"
        )

    # Recorded frames must equal the config nsteps attr (catches stale/mixed
    # record files from a previous run of a different duration).
    if n_steps != nsteps_artifact:
        raise ValueError(
            f"recorded frames ({n_steps}) != config.h5 nsteps ({nsteps_artifact}) — "
            "stale/mixed-run records; re-run the pipeline"
        )
    stf_values = stf_values[:n_steps]
    if len(stf_values) < n_steps:
        raise ValueError(
            f"config.h5 STF shorter ({len(stf_values)}) than recorded frames ({n_steps})"
        )

    return {
        "vp_m_s": vp,
        "vs_m_s": vs,
        "density_kg_m3": rho,
        "source_xyz": source_xyz,
        "output_dt_s": output_dt_s,
        "stf": stf_values,
    }


def interior_bounds_from_config(config_h5: str, model_h5: str) -> np.ndarray:
    """Per-axis [lo, hi] bounds of the non-PML interior, derived from artifacts.

    PML thicknesses are stored in config.h5 /simulation attrs as element
    counts (pml_xmin, ...). Element size = real mesh extent / n_elements, and
    the real mesh extent must come from model.h5 cell coords (0..18000 here) —
    NOT from the Green's-function tile coords, which are cropped to the
    recording region and would give a wrong element size. Deriving the
    interior from the CURRENT run's geometry (instead of a hardcoded box)
    keeps the receiver sample inside the true physical interior — a stale box
    silently mixes PML-region vertices into the comparison and drags the
    correlation down.
    """
    with h5py.File(config_h5, "r") as f:
        sim = f["/simulation"]
        n_elem = np.array(
            [
                _to_int(np.asarray(sim.attrs["nx_elements"]), "nx_elements"),
                _to_int(np.asarray(sim.attrs["ny_elements"]), "ny_elements"),
                _to_int(np.asarray(sim.attrs["nz_elements"]), "nz_elements"),
            ]
        )
        pml_min = np.array(
            [
                _to_int(np.asarray(sim.attrs[f"pml_{axis}min"]), f"pml_{axis}min")
                for axis in ("x", "y", "z")
            ]
        )
        pml_max = np.array(
            [
                _to_int(np.asarray(sim.attrs[f"pml_{axis}max"]), f"pml_{axis}max")
                for axis in ("x", "y", "z")
            ]
        )
    with h5py.File(model_h5, "r") as f:
        # /field/cell/coords is (n_cell, NGLL, NGLL, NGLL, 3) — per-cell GLL
        # nodes, so reduce over the node axes to get per-axis domain extent.
        cell_coords = np.asarray(f["/field/cell/coords"])
        extent = cell_coords.max(axis=(0, 1, 2, 3)) - cell_coords.min(axis=(0, 1, 2, 3))
    element_size = extent / n_elem
    bounds = np.stack([pml_min * element_size, extent - pml_max * element_size], axis=1)
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError(f"config.h5 PML covers the whole domain ({bounds}) — bad interior box")
    return bounds


def _nearest_node_indices(coords, points) -> list:
    """Index of the nearest recorded GLL node for each fixed physical point.

    The analytical solution is evaluated at the NEAREST NODE'S true
    coordinates (not the nominal point), so a coarse mesh only changes which
    node is sampled, never injecting an h-dependent position/phase error.
    """
    coords = np.asarray(coords, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"--fixed-receivers must be an (N,3) array, got {points.shape}")
    try:
        from scipy.spatial import cKDTree  # scipy is present in the project venv

        _, idx = cKDTree(coords).query(points, k=1)
        return list(map(int, idx))
    except ImportError:
        # Fallback without scipy: chunked brute force.
        best: list[int] = []
        for point in points:
            delta = coords - point
            best.append(int(np.argmin(np.einsum("ij,ij->i", delta, delta))))
        return list(map(int, np.unique(best)))


def load_sem_tiles(greenfun_dir: str) -> dict:
    """Load tile metadata and merge coordinates without loading field tensors."""
    tiles = sorted(glob.glob(os.path.join(greenfun_dir, "tile_*.h5")))
    if not tiles:
        raise FileNotFoundError(f"No tile_*.h5 in {greenfun_dir}")

    all_coords = []
    tile_ranges = []
    dt_s = None
    n_steps = None
    displacement_dtype = None
    vertex_offset = 0
    for tp in tiles:
        with h5py.File(tp, "r") as f:
            c = np.asarray(f["/mesh/gll_node_coords"])
            displacement = f["/field/displacement_tensor"]
            if n_steps is None:
                n_steps = displacement.shape[0]
                displacement_dtype = displacement.dtype
            if dt_s is None:
                times = np.asarray(f["/time/t"])
                dt_s = _to_float(times[1] - times[0], "time/t.dt") if len(times) > 1 else 0.01
            all_coords.append(c)
            tile_ranges.append((tp, vertex_offset, vertex_offset + len(c)))
            vertex_offset += len(c)

    return {
        "coords": np.concatenate(all_coords, axis=0),
        "dt_s": dt_s or 0.01,
        "n_steps": n_steps,
        "displacement_dtype": displacement_dtype,
        "tile_ranges": tile_ranges,
    }


def load_sampled_displacement(sem: dict, sample_indices: list[int]) -> np.ndarray:
    """Load displacement traces only for selected global vertex indices."""
    indices = np.asarray(sample_indices, dtype=np.int64)
    if len(indices) == 0:
        return np.empty((sem["n_steps"], 0, 3, 3), dtype=sem["displacement_dtype"])
    if indices.min() < 0 or indices.max() >= len(sem["coords"]):
        raise IndexError("sample vertex index outside merged tile coordinates")

    sampled = np.empty((sem["n_steps"], len(indices), 3, 3), dtype=sem["displacement_dtype"])
    for tile_path, vertex_start, vertex_stop in sem["tile_ranges"]:
        sample_positions = np.flatnonzero((indices >= vertex_start) & (indices < vertex_stop))
        if len(sample_positions) == 0:
            continue

        local_indices = indices[sample_positions] - vertex_start
        unique_local_indices, inverse = np.unique(local_indices, return_inverse=True)
        with h5py.File(tile_path, "r") as f:
            unique_displacement = f["/field/displacement_tensor"][:, unique_local_indices, :, :]
            sampled[:, sample_positions, :, :] = unique_displacement[:, inverse, :, :]
    return sampled


def parse_args() -> dict:
    """Parse command-line arguments."""
    args = {
        "source": None,
        "fullspace": False,
        "greenfun_dir": None,
        "config_h5": None,
        "model_h5": None,
        "interior_box": None,
        "fixed_receivers": None,
    }
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--source" and i + 3 < len(sys.argv):
            args["source"] = np.array(
                [
                    _to_float(sys.argv[i + 1], "--source x"),
                    _to_float(sys.argv[i + 2], "--source y"),
                    _to_float(sys.argv[i + 3], "--source z"),
                ],
                dtype=np.float64,
            )
            i += 4
        elif sys.argv[i] == "--fullspace":
            args["fullspace"] = True
            i += 1
        elif sys.argv[i] == "--config-h5" and i + 1 < len(sys.argv):
            args["config_h5"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--model-h5" and i + 1 < len(sys.argv):
            args["model_h5"] = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--interior-box" and i + 2 < len(sys.argv):
            args["interior_box"] = (
                _to_float(sys.argv[i + 1], "--interior-box lo"),
                _to_float(sys.argv[i + 2], "--interior-box hi"),
            )
            i += 3
        elif sys.argv[i] == "--fixed-receivers" and i + 1 < len(sys.argv):
            args["fixed_receivers"] = sys.argv[i + 1]
            i += 2
        elif not args["greenfun_dir"]:
            args["greenfun_dir"] = sys.argv[i]
            i += 1
        else:
            i += 1
    return args


def best_fit_scale(sem_samps: np.ndarray, ana_samps: np.ndarray) -> float:
    """Least-squares SEM = scale * analytical scale fitting.

    Returns scale s minimizing ||sem - s*ana|| (reported alongside the raw
    metrics; makes the documented ~3x SEM amplitude factor explicit).
    """
    denom = _to_float(np.dot(ana_samps, ana_samps), "scale denom")
    if denom < 1e-300:
        return 1.0
    return _to_float(np.dot(sem_samps, ana_samps) / denom, "scale")


def amplitude_scale_is_acceptable(scale: float) -> bool:
    """Return whether the absolute SEM amplitude agrees with the reference."""
    return MIN_ACCEPTABLE_SCALE <= scale <= MAX_ACCEPTABLE_SCALE


def scale_fitted_relative_l2(
    sem_samples: np.ndarray, analytical_samples: np.ndarray
) -> tuple[float, float]:
    """Fit analytical amplitude to SEM and return a scale-invariant relative L2."""
    scale = best_fit_scale(sem_samples, analytical_samples)
    return scale, relative_l2_error(sem_samples, scale * analytical_samples)


def run(
    greenfun_dir,
    source_override=None,
    fullspace=False,
    config_h5=None,
    model_h5=None,
    interior_box=None,
    fixed_receivers=None,
):
    """Run analytical comparison; returns exit code."""
    print(f"Loading SEM tiles from {greenfun_dir} ...")
    sem = load_sem_tiles(greenfun_dir)
    n_steps = sem["n_steps"]
    dt_s = sem["dt_s"]
    coords = sem["coords"]
    n_vertex = coords.shape[0]
    total_s = n_steps * dt_s
    print(f"  n_steps={n_steps}, dt_s={dt_s:.4f}s, n_vertex={n_vertex}, duration={total_s:.2f}s")

    # Read SEM parameters from the run artifacts (source of truth).
    config_path, model_path = find_artifacts(greenfun_dir, config_h5, model_h5)
    print(f"  SEM params from {os.path.basename(config_path)} + {os.path.basename(model_path)}")
    params = load_sem_parameters(config_path, model_path, n_steps)
    vp = params["vp_m_s"]
    vs = params["vs_m_s"]
    rho = params["density_kg_m3"]
    source_xyz = params["source_xyz"]
    stf = params["stf"]
    output_dt_s = params["output_dt_s"]

    # Hard consistency assertion: tile dt must equal config output_dt_s.
    if not np.isclose(dt_s, output_dt_s, rtol=1e-9, atol=1e-12):
        raise RuntimeError(
            f"tile dt={dt_s} != config.h5 output_dt_s={output_dt_s} — refusing to compare "
            "(STF and records would be misaligned)"
        )

    # Optional source override — a conflicting value is an ERROR (the SEM
    # source in config.h5 is authoritative; ambiguity must not be silently
    # resolved, it corrupts the comparison).
    if source_override is not None:
        if not np.allclose(source_override, source_xyz, rtol=1e-9, atol=1e-6):
            raise ValueError(
                f"--source {source_override} != config.h5 source {source_xyz} — "
                "refusing ambiguous comparison; omit --source to use the SEM source"
            )
        source_xyz = source_override

    print(f"  source: {source_xyz}, mode: {'fullspace' if fullspace else 'halfspace'}")
    print(f"  material: vp={vp:.1f}, vs={vs:.1f}, rho={rho:.1f} (from model.h5)")

    # Dominant frequency of the SEM STF (for distance grouping below).
    stf_centered = stf - _to_float(np.mean(stf), "stf mean")
    spec = np.abs(np.fft.rfft(stf_centered))
    freqs = np.fft.rfftfreq(len(stf), dt_s)
    f_dom = _to_float(freqs[1:][np.argmax(spec[1:])], "stf dominant frequency")
    lambda_s = vs / f_dom
    print(f"  SEM STF dominant frequency: {f_dom:.3f} Hz (lambda_s={lambda_s:.0f} m)")

    # Subsample receivers from interior (exclude PML region). The interior is
    # DERIVED from config.h5 PML thicknesses x current element size, so the
    # sample can never silently include PML-region vertices after a geometry
    # change (a hardcoded box caused exactly that: [3000,15000] m included
    # ~2.25 km of real PML on every side of the 24³/PML-7 mesh).
    if fixed_receivers is not None:
        points = np.asarray(np.load(fixed_receivers), dtype=np.float64)
        sample_indices = _nearest_node_indices(coords, points)
        print(
            f"  {len(points)} fixed receivers -> {len(set(sample_indices))} unique "
            "nearest recorded GLL nodes"
        )
        stride = 0
    elif fullspace:
        if interior_box is not None:
            # Explicit common box (grid-convergence runs: same receiver
            # sampling region in meters across meshes of different element
            # sizes, so mean_corr is not tuned by per-grid interior width).
            lo = np.array([interior_box[0]] * 3)
            hi = np.array([interior_box[1]] * 3)
        else:
            bounds = interior_bounds_from_config(config_path, model_path)
            lo, hi = bounds[:, 0], bounds[:, 1]
        interior_mask = (
            (coords[:, 0] >= lo[0])
            & (coords[:, 0] <= hi[0])
            & (coords[:, 1] >= lo[1])
            & (coords[:, 1] <= hi[1])
            & (coords[:, 2] >= lo[2])
            & (coords[:, 2] <= hi[2])
        )
        interior_indices = list(map(int, np.where(interior_mask)[0]))
        n_interior = len(interior_indices)
        print(
            f"  interior vertices (non-PML): {n_interior}/{n_vertex} "
            f"(box {lo[0]:.0f}..{hi[0]:.0f} m)"
        )
        stride = max(1, n_interior // 50)
        sample_indices = interior_indices[::stride][:50]
    else:
        stride = max(1, n_vertex // 100)
        sample_indices = list(range(0, n_vertex, stride))[:100]
    if stride:
        print(f"  sampling {len(sample_indices)} receivers (stride={stride})")
    else:
        print(f"  sampling {len(sample_indices)} receivers (fixed physical points)")
    sampled_displacement = load_sampled_displacement(sem, sample_indices)

    all_corrs = []
    all_l2 = []
    all_sem = []
    all_ana = []
    # Distance-grouped diagnostics (in S wavelengths).
    dist_bins = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, np.inf)]
    bin_corrs = {b: [] for b in dist_bins}

    for force_dir in range(3):
        f_label = FORCE_LABELS[force_dir]
        f_corrs = []
        f_l2 = []

        for sample_position, pt_idx in enumerate(sample_indices):
            receiver = np.asarray(coords[pt_idx], dtype=np.float64)

            if fullspace:
                cutoff = n_steps
            else:
                try:
                    t_refl = first_surface_reflected_arrival_time_s(receiver, source_xyz, vp)
                    cutoff = min(n_steps, int(t_refl / dt_s) + 1)
                except Exception:
                    continue
                if cutoff < 10:
                    continue

            try:
                analytical = stokes_displacement_green_tensor(
                    receiver, source_xyz, force_dir, vp, vs, rho, stf, dt_s
                )
            except Exception:
                continue

            r = _to_float(np.linalg.norm(receiver - source_xyz), "receiver distance")
            recv_corrs = []

            for disp_comp in range(3):
                sem_wave = sampled_displacement[:cutoff, sample_position, disp_comp, force_dir]
                ana_wave = analytical[:cutoff, disp_comp]

                c = wavelet_correlation(sem_wave, ana_wave)
                e = relative_l2_error(sem_wave, ana_wave)
                recv_corrs.append(c)
                f_corrs.append(c)
                f_l2.append(e)
                all_corrs.append(c)
                all_l2.append(e)
                all_sem.append(sem_wave)
                all_ana.append(ana_wave)

            # Distance bin by S wavelength (f_dom from the SEM STF).
            for lo, hi in dist_bins:
                if lo <= r / lambda_s < hi:
                    bin_corrs[(lo, hi)].append(_to_float(np.mean(recv_corrs), "bin corr"))
                    break

        if f_corrs:
            print(
                f"  Force {f_label}: n={len(f_corrs)}, mean_corr={np.mean(f_corrs):.4f}, "
                f"mean_l2={np.mean(f_l2):.4f}"
            )
        else:
            print(f"  Force {f_label}: no valid comparisons")

    if not all_corrs:
        print("\nFAILED: no valid comparisons")
        return 1

    # Aggregate metrics.
    mean_corr = _to_float(np.mean(all_corrs), "overall corr")
    mean_l2 = _to_float(np.mean(all_l2), "overall l2")
    sem_flat = np.concatenate([w.reshape(-1) for w in all_sem])
    ana_flat = np.concatenate([w.reshape(-1) for w in all_ana])
    scale, fitted_l2 = scale_fitted_relative_l2(sem_flat, ana_flat)

    print(f"\n{'=' * 60}")
    print(f" OVERALL: mean_corr={mean_corr:.4f}, mean_l2={mean_l2:.4f}")
    print(f" best-fit scale (SEM/analytical)={scale:.3f}, scale-fitted L2={fitted_l2:.4f}")
    print(f"{'=' * 60}")

    print("\n  correlation vs receiver distance (in S wavelengths, f0 from SEM STF):")
    for (lo, hi), cs in bin_corrs.items():
        if cs:
            tag = "inf" if hi == np.inf else f"{hi:.1f}"
            print(f"    r/lambda_s in [{lo:.1f}, {tag}): n={len(cs)}, mean_corr={np.mean(cs):.4f}")

    if not amplitude_scale_is_acceptable(scale):
        print(
            f"\nFAILED: best-fit scale={scale:.3f} outside "
            f"[{MIN_ACCEPTABLE_SCALE:.1f}, {MAX_ACCEPTABLE_SCALE:.1f}]"
        )
        return 1

    if mean_corr >= 0.95:
        print(f"\nPASSED: mean_corr={mean_corr:.4f} >= 0.95")
        return 0
    elif mean_corr >= 0.80:
        print(f"\nWARNING: mean_corr={mean_corr:.4f} in [0.80, 0.95)")
        return 0
    else:
        print(f"\nFAILED: mean_corr={mean_corr:.4f} < 0.80")
        return 1


if __name__ == "__main__":
    args = parse_args()
    if not args["greenfun_dir"]:
        print(
            f"Usage: {sys.argv[0]} <greenfun_dir> [--source x y z] [--fullspace] "
            "[--interior-box LO HI] [--fixed-receivers FILE]"
        )
        sys.exit(1)
    sys.exit(
        run(
            args["greenfun_dir"],
            args["source"],
            args["fullspace"],
            args["config_h5"],
            args["model_h5"],
            args["interior_box"],
            args["fixed_receivers"],
        )
    )
