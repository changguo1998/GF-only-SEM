#!/usr/bin/env python3
"""pyfk-based analytical Green's function comparison for elastic media.

Uses pyfk 0.2.0 (Python 3.9) to compute exact FK Green's functions
for a homogeneous half-space or layered medium. The pyfk output
(6 raw FK kernels for SF source) is post-processed with the correct
radiation pattern and rotated from (Z,R,T) to Cartesian (x,y,z).

Requires: .venv-pyfk/ with pyfk, numpy, scipy, h5py installed.

Usage:
    .venv-pyfk/bin/python analytical_pyfk_compare.py <greenfun_dir>
"""

import sys
import os
import glob

import numpy as np
import h5py

# pyfk imports
from pyfk import Config, SeisModel, SourceModel, calculate_gf


# ── pyfk post-processing ────────────────────────────────────────────────


def sf_radiation_weights(az_deg: float, stk_deg: float, dip_deg: float) -> np.ndarray:
    """Compute radiation pattern weights for a single-force source.

    Returns rad[4][3] array; for SF only rad[0] and rad[1] are non-zero.
    rad[0] = Z0 coefficients for (V,R,T)
    rad[1] = Z1 coefficients for (V,R,T)

    Parameters
    ----------
    az_deg : receiver azimuth measured CW from North [degrees].
    stk_deg : force strike [degrees].
    dip_deg : force dip from horizontal [degrees].
    """
    az_rad = np.deg2rad(az_deg)
    stk_rad = np.deg2rad(stk_deg)
    dip_rad = np.deg2rad(dip_deg)

    rad = np.zeros((4, 3))
    # Z0 (m=0): depends only on dip
    rad[0, 0] = -np.sin(dip_rad)  # vertical
    rad[0, 1] = rad[0, 0]  # radial
    rad[0, 2] = 0.0  # transverse (zero for m=0)
    # Z1 (m=1): depends on az and dip
    rad[1, 0] = np.cos(dip_rad) * np.cos(az_rad - stk_rad)  # vertical
    rad[1, 1] = rad[1, 0]  # radial
    rad[1, 2] = np.cos(dip_rad) * np.sin(az_rad - stk_rad)  # transverse
    return rad


def pyfk_to_cartesian(
    pyfk_traces: list,
    receiver_xyz_m: np.ndarray,
    source_xyz_m: np.ndarray,
    force_stk_deg: float,
    force_dip_deg: float,
    dt_s: float,
) -> np.ndarray:
    """Convert pyfk 6-trace FK output to Cartesian displacement.

    Parameters
    ----------
    pyfk_traces : list of 6 Obspy Trace objects (ZVF,RVF,THF,ZHF,RHF,THF).
    receiver_xyz_m : (3,) receiver position [m].
    source_xyz_m : (3,) source position [m].
    force_stk_deg : force strike [degrees CW from North].
    force_dip_deg : force dip [degrees from horizontal].
    dt_s : time step [s] (for trace length verification).

    Returns
    -------
    displacement : (nt, 3) ndarray — displacement [ux, uy, uz] in meters,
        where x=East, y=North, z=Down (SEM convention).
    """
    # Get trace data
    nt = len(pyfk_traces[0].data)
    kernel = np.zeros((6, nt))
    for i in range(6):
        kernel[i, :] = pyfk_traces[i].data

    # Receiver azimuth
    dx = receiver_xyz_m[0] - source_xyz_m[0]
    dy = receiver_xyz_m[1] - source_xyz_m[1]
    r_horiz = np.sqrt(dx**2 + dy**2)
    if r_horiz > 1e-6:
        az_deg = np.rad2deg(np.arctan2(dx, dy))  # CW from North
    else:
        az_deg = 0.0

    # Radiation weights
    rad = sf_radiation_weights(az_deg, force_stk_deg, force_dip_deg)

    # Combine kernels with radiation weights
    # trace 0=ZVF, 1=RVF, 2=THF, 3=ZHF, 4=RHF, 5=THF(m=1)
    disp_V = rad[0, 0] * kernel[0] + rad[1, 0] * kernel[3]  # vertical (up)
    disp_R = rad[0, 1] * kernel[1] + rad[1, 1] * kernel[4]  # radial (away)
    disp_T = rad[0, 2] * kernel[2] + rad[1, 2] * kernel[5]  # transverse (CW)

    # Rotate to Cartesian (x=East, y=North, z=Down)
    if r_horiz > 1e-6:
        cos_az = dy / r_horiz  # cos(az) = N component of radial
        sin_az = dx / r_horiz  # sin(az) = E component of radial
        ux = disp_R * sin_az - disp_T * cos_az  # East
        uy = disp_R * cos_az + disp_T * sin_az  # North
    else:
        ux = np.zeros(nt)
        uy = np.zeros(nt)

    uz = -disp_V  # pyfk V=up, SEM z=down

    return np.column_stack([ux, uy, uz])


# ── SEM data loading ─────────────────────────────────────────────────────


def load_sem_tiles(greenfun_dir: str) -> dict:
    """Load and merge all Green's function tiles."""
    tiles = sorted(glob.glob(os.path.join(greenfun_dir, "tile_*.h5")))
    if not tiles:
        raise FileNotFoundError(f"No tile_*.h5 in {greenfun_dir}")

    all_disp = []
    all_coords = []
    dt_s = None

    for tp in tiles:
        with h5py.File(tp, "r") as f:
            d = np.asarray(f["/field/displacement_tensor"])  # (nt, nv, 3, 3)
            c = np.asarray(f["/mesh/gll_node_coords"])  # (nv, 3)
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


# ── Waveform metrics ─────────────────────────────────────────────────────


def wavelet_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation."""
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def relative_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 norm error."""
    dn = float(np.linalg.norm(a - b))
    mn = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    if mn < 1e-30:
        return 0.0
    return dn / mn


# ── Main comparison ──────────────────────────────────────────────────────


def run(greenfun_dir: str):
    """Run pyfk comparison; returns exit code."""
    print(f"Loading SEM tiles from {greenfun_dir} ...")
    sem = load_sem_tiles(greenfun_dir)
    n_steps = sem["n_steps"]
    dt_s = sem["dt_s"]
    coords = sem["coords"]
    n_vertex = coords.shape[0]
    total_s = n_steps * dt_s

    print(f"  n_steps={n_steps}, dt_s={dt_s:.4f}s, n_vertex={n_vertex}")

    # Halfspace model: Vp=5 km/s, Vs=3 km/s, rho=2.7 g/cm³
    fk_model = np.array([[0.0, 3.0, 5.0, 2.7, 100.0, 200.0]])
    seismodel = SeisModel(fk_model)

    # Source at 278m depth
    source_xyz = np.array([5278.0, 5278.0, 278.0])

    # Compute pyfk output once (same FK kernels for all force directions)
    # Use a moderate distance receiver to compute FK kernels
    pyfk_config = Config(
        model=seismodel,
        source=SourceModel(sdep=0.278, srcType="sf"),
        receiver_distance=[1.0],  # dummy, we need FK kernels not final seismograms
        rdep=0.0,
        npt=n_steps,
        dt=dt_s,
        samples_before_first_arrival=50,
    )

    print("  Computing pyfk FK kernels ...")
    pyfk_streams = calculate_gf(pyfk_config)
    print(f"  Got {len(pyfk_streams[0])} traces per receiver")

    # pyfk output at a fixed distance — we need per-receiver kernels.
    # pyfk only supports fixed receiver distances, not per-receiver FK.
    # For a homogeneous halfspace, the FK kernels at distance r are
    # related to kernels at distance r0 by a simple geometric scaling
    # and time shift. But for verification, we'll use pyfk at a SINGLE
    # distance and compare with SEM at a matching receiver.

    # For now: find a receiver at approximately 1 km distance
    dists = np.sqrt(np.sum((coords - source_xyz) ** 2, axis=1))
    target_dist = 1000.0  # 1 km
    best_idx = np.argmin(np.abs(dists - target_dist))
    best_dist = dists[best_idx]

    print(f"  Closest receiver to {target_dist}m: idx={best_idx}, dist={best_dist:.0f}m")

    # Rerun pyfk at the actual distance
    pyfk_config_actual = Config(
        model=seismodel,
        source=SourceModel(sdep=0.278, srcType="sf"),
        receiver_distance=[best_dist / 1000.0],  # convert to km
        rdep=float(coords[best_idx, 2]) / 1000.0,  # receiver depth in km
        npt=n_steps,
        dt=dt_s,
        samples_before_first_arrival=50,
    )
    pyfk_actual = calculate_gf(pyfk_config_actual)

    # Convert pyfk to Cartesian for 3 force directions
    fd_params = [
        (0, 0.0, 0.0, "fx"),  # force North (x), stk=0, dip=0
        (1, 90.0, 0.0, "fy"),  # force East (y), stk=90, dip=0
        (2, 0.0, 90.0, "fz"),  # force vertical (z), stk=0, dip=90
    ]

    receiver = np.asarray(coords[best_idx], dtype=np.float64)
    print(f"\n  Receiver: {receiver}")

    all_corrs = []
    all_l2 = []

    for fd_idx, stk, dip, label in fd_params:
        pyfk_cart = pyfk_to_cartesian(pyfk_actual[0], receiver, source_xyz, stk, dip, dt_s)

        for comp in range(3):
            sem_wave = sem["displacement"][:, best_idx, comp, fd_idx]
            pyfk_wave = pyfk_cart[:, comp]

            # Window: avoid leading/trailing zeros
            mask = np.abs(sem_wave) > 1e-10 * np.max(np.abs(sem_wave))
            if np.sum(mask) < 20:
                continue
            start = max(0, np.argmax(mask) - 10)
            end = min(n_steps, len(mask) - np.argmax(mask[::-1]) + 10)

            c = wavelet_correlation(sem_wave[start:end], pyfk_wave[start:end])
            e = relative_l2_error(sem_wave[start:end], pyfk_wave[start:end])

            all_corrs.append(c)
            all_l2.append(e)
            comp_label = ["x", "y", "z"][comp]
            print(f"    {label}→{comp_label}: corr={c:.4f}, l2={e:.4f}")

    if not all_corrs:
        print("\nFAILED: no valid comparisons")
        return 1

    mean_c = float(np.mean(all_corrs))
    mean_l2 = float(np.mean(all_l2))
    print(f"\n{'=' * 60}")
    print(f" OVERALL: mean_corr={mean_c:.4f}, mean_l2={mean_l2:.4f}")
    print(f"{'=' * 60}")

    if mean_c >= 0.95:
        print(f"\nPASSED: mean_corr={mean_c:.4f} >= 0.95")
        return 0
    elif mean_c >= 0.80:
        print(f"\nWARNING: mean_corr={mean_c:.4f} in [0.80, 0.95)")
        return 0
    else:
        print(f"\nFAILED: mean_corr={mean_c:.4f} < 0.80")
        return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <greenfun_dir>")
        sys.exit(1)
    sys.exit(run(sys.argv[1]))
