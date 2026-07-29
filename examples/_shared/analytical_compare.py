#!/usr/bin/env python3
"""Compare SEM Green's function tiles against analytical full-space Stokes solution.

For a full-space setup (all PML boundaries, source away from boundaries),
the full-space Stokes solution is the exact reference — no free-surface
reflections to exclude. Comparison uses the entire waveform.

For a half-space setup (free surface at z=0), comparison is time-windowed
to before the first surface-reflected phase.

Usage:
    python3 analytical_compare.py <greenfun_dir> [--source x y z] [--fullspace]
"""

import sys
import os
import glob

import numpy as np
import h5py

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from analytical_green import (
    make_ricker_stf,
    stokes_displacement_green_tensor,
    wavelet_correlation,
    relative_l2_error,
    first_surface_reflected_arrival_time_s,
)

FORCE_LABELS = ["x", "y", "z"]


def load_sem_tiles(greenfun_dir):
    """Load and merge all Green's function tiles."""
    tiles = sorted(glob.glob(os.path.join(greenfun_dir, "tile_*.h5")))
    if not tiles:
        raise FileNotFoundError(f"No tile_*.h5 in {greenfun_dir}")

    all_disp = []
    all_coords = []
    dt_s = None

    for tp in tiles:
        with h5py.File(tp, "r") as f:
            d = np.asarray(f["/field/displacement_tensor"])
            c = np.asarray(f["/mesh/gll_node_coords"])
            if dt_s is None:
                times = np.asarray(f["/time/t"])
                dt_s = float(times[1] - times[0]) if len(times) > 1 else 0.01
            all_disp.append(d)
            all_coords.append(c)

    return {
        "displacement": np.concatenate(all_disp, axis=1),
        "coords": np.concatenate(all_coords, axis=0),
        "dt_s": dt_s or 0.01,
        "n_steps": all_disp[0].shape[0],
    }


def parse_args():
    """Parse command-line arguments."""
    args = {"source": None, "fullspace": False, "greenfun_dir": None}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--source" and i + 3 < len(sys.argv):
            args["source"] = np.array(
                [float(sys.argv[i + 1]), float(sys.argv[i + 2]), float(sys.argv[i + 3])],
                dtype=np.float64,
            )
            i += 4
        elif sys.argv[i] == "--fullspace":
            args["fullspace"] = True
            i += 1
        elif not args["greenfun_dir"]:
            args["greenfun_dir"] = sys.argv[i]
            i += 1
        else:
            i += 1
    return args


def run(greenfun_dir, source_xyz=None, fullspace=False):
    """Run analytical comparison; returns exit code."""
    print(f"Loading SEM tiles from {greenfun_dir} ...")
    sem = load_sem_tiles(greenfun_dir)
    n_steps = sem["n_steps"]
    dt_s = sem["dt_s"]
    coords = sem["coords"]
    n_vertex = coords.shape[0]
    total_s = n_steps * dt_s
    print(f"  n_steps={n_steps}, dt_s={dt_s:.4f}s, n_vertex={n_vertex}, duration={total_s:.2f}s")

    # Material properties (same for all current models)
    vp = 5000.0
    vs = 3000.0
    rho = 2700.0

    # Source location
    if source_xyz is None:
        source_xyz = np.array([5278.0, 5278.0, 278.0], dtype=np.float64)  # halfspace default
    print(f"  source: {source_xyz}, mode: {'fullspace' if fullspace else 'halfspace'}")

    _, stf = make_ricker_stf(1.0, 2.0, 1.0e20, dt_s, total_s)
    stf = np.asarray(stf, dtype=np.float64)

    # Subsample receivers from interior (exclude PML region)
    if fullspace:
        interior_mask = (
            (coords[:, 0] >= 3000.0)
            & (coords[:, 0] <= 15000.0)
            & (coords[:, 1] >= 3000.0)
            & (coords[:, 1] <= 15000.0)
            & (coords[:, 2] >= 3000.0)
            & (coords[:, 2] <= 15000.0)
        )
        interior_indices = list(map(int, np.where(interior_mask)[0]))
        n_interior = len(interior_indices)
        print(f"  interior vertices (non-PML): {n_interior}/{n_vertex}")
        stride = max(1, n_interior // 50)
        sample_indices = interior_indices[::stride][:50]
    else:
        stride = max(1, n_vertex // 100)
        sample_indices = list(range(0, n_vertex, stride))[:100]
    print(f"  sampling {len(sample_indices)} receivers (stride={stride})")

    all_corrs = []
    all_l2 = []

    for force_dir in range(3):
        f_label = FORCE_LABELS[force_dir]
        f_corrs = []
        f_l2 = []

        for pt_idx in sample_indices:
            receiver = np.asarray(coords[pt_idx], dtype=np.float64)

            if fullspace:
                # Full-space: use entire waveform
                cutoff = n_steps
            else:
                # Half-space: time-window before first surface reflection
                t_refl = first_surface_reflected_arrival_time_s(receiver, source_xyz, vp)
                cutoff = min(n_steps, int(t_refl / dt_s) + 1)
                if cutoff < 10:
                    continue

            try:
                analytical = stokes_displacement_green_tensor(
                    receiver, source_xyz, force_dir, vp, vs, rho, stf, dt_s
                )
            except Exception:
                continue

            for disp_comp in range(3):
                sem_wave = sem["displacement"][:cutoff, pt_idx, disp_comp, force_dir]
                ana_wave = analytical[:cutoff, disp_comp]

                c = wavelet_correlation(sem_wave, ana_wave)
                e = relative_l2_error(sem_wave, ana_wave)
                f_corrs.append(c)
                f_l2.append(e)
                all_corrs.append(c)
                all_l2.append(e)

        if f_corrs:
            print(
                f"  Force {f_label}: n={len(f_corrs)}, mean_corr={np.mean(f_corrs):.4f}, mean_l2={np.mean(f_l2):.4f}"
            )
        else:
            print(f"  Force {f_label}: no valid comparisons")

    if not all_corrs:
        print("\nFAILED: no valid comparisons")
        return 1

    mean_corr = float(np.mean(all_corrs))
    mean_l2 = float(np.mean(all_l2))
    print(f"\n{'=' * 60}")
    print(f" OVERALL: mean_corr={mean_corr:.4f}, mean_l2={mean_l2:.4f}")
    print(f"{'=' * 60}")

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
        print(f"Usage: {sys.argv[0]} <greenfun_dir> [--source x y z] [--fullspace]")
        sys.exit(1)
    sys.exit(run(args["greenfun_dir"], args["source"], args["fullspace"]))
