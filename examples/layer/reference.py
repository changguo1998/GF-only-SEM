#!/usr/bin/env python3
"""PyFK reference waveform for a layered elastic half-space.

Computes the displacement Green tensor (or synthetic seismogram) for a
1D layered model defined in ``config.py``.  Requires PyFK (Python 3.8/3.9)
installed in ``examples/layer/.pyfk-venv/``.

Two output modes:

* **Green function** (default) — ``displacement`` in m/N.
* **Synthetic seismogram** (``--ricker-freq``) — displacement convolved
  with a Ricker wavelet for the given peak frequency.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Make the project root importable for consistency with the halfspace example.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = (_SCRIPT_DIR / "../..").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from greenfun.index_cache import load_or_rebuild_index  # noqa: E402

# Layered model parameters
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402  # isort: skip
    LAYER_MODEL,
    FORCE_AMPLITUDE,
    DK,
    SMTH,
    PMIN,
    PMAX,
    KMAX,
    SAMPLES_BEFORE_FIRST_ARRIVAL,
)

_FORCE_DIRECTIONS = ("x", "y", "z")


def _require_pyfk() -> dict[str, Any]:
    """Import PyFK lazily and return its key classes/functions."""
    import importlib.util

    for mod_name in ("pyfk", "obspy"):
        if importlib.util.find_spec(mod_name) is None:
            raise RuntimeError(
                f"{mod_name} is required.  Run this script with the PyFK environment:\n"
                f"  examples/layer/.pyfk-venv/bin/python ..."
            )
    from obspy import UTCDateTime  # type: ignore[import-not-found]
    from pyfk import Config, SeisModel, SourceModel, calculate_gf  # type: ignore[import-not-found]
    from pyfk.sync.sync import sync_calculate_gf  # type: ignore[import-not-found]

    return {
        "Config": Config,
        "SeisModel": SeisModel,
        "SourceModel": SourceModel,
        "UTCDateTime": UTCDateTime,
        "calculate_gf": calculate_gf,
        "sync_calculate_gf": sync_calculate_gf,
    }


def _station_azimuth_deg(dx_m: float, dy_m: float) -> float:
    return math.degrees(math.atan2(dx_m, dy_m)) % 360.0


def _force_orientation(force_direction: str) -> tuple[float, float]:
    """Return PyFK mechanism ``(strike, dip)`` for a Cartesian force component."""
    mapper = {"x": (90.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 90.0)}
    try:
        return mapper[force_direction]
    except KeyError:
        raise ValueError(f"Unsupported force direction {force_direction!r}")


def _sync_trace_to_time(
    trace: Any, target_time: npt.NDArray[np.float64], utc_epoch: Any
) -> npt.NDArray[np.float64]:
    """Resample an obspy Trace to *target_time* and convert cm → m."""
    start_s = float(trace.stats.starttime - utc_epoch)
    tr_time = start_s + np.arange(trace.stats.npts, dtype=np.float64) * float(trace.stats.delta)
    data_m = np.asarray(trace.data, dtype=np.float64) * 0.01
    return np.interp(target_time, tr_time, data_m, left=0.0, right=0.0)


def _urt_to_xyz_down(
    up_m: npt.NDArray[np.float64],
    radial_m: npt.NDArray[np.float64],
    transverse_m: npt.NDArray[np.float64],
    azimuth_deg: float,
) -> npt.NDArray[np.float64]:
    """Convert up-radial-transverse (ZRT) to Cartesian (x, y, z) with z-down."""
    az_rad = math.radians(azimuth_deg)
    displacement = np.empty((up_m.size, 3), dtype=np.float64)
    displacement[:, 0] = radial_m * math.sin(az_rad) + transverse_m * math.cos(az_rad)
    displacement[:, 1] = radial_m * math.cos(az_rad) - transverse_m * math.sin(az_rad)
    displacement[:, 2] = -up_m
    return displacement


def compute_green_tensor(
    source_xyz_m: npt.NDArray[np.float64],
    receiver_xyz_m: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    force_amplitude: float = FORCE_AMPLITUDE,
    samples_before_first_arrival: int = SAMPLES_BEFORE_FIRST_ARRIVAL,
) -> npt.NDArray[np.float64]:
    """Displacement Green tensor ``u[i, j]`` (m/N) via PyFK.

    Parameters
    ----------
    source_xyz_m : (3,) array
        Source location [m].
    receiver_xyz_m : (3,) array
        Receiver location [m].
    time : (nt,) array
        Time samples [s].
    force_amplitude : float
        PyFK single-force amplitude.
    samples_before_first_arrival : int
        Samples prepended before first arrival.

    Returns
    -------
    G : (nt, 3, 3) array
        Green tensor.  ``G[:, i, j]`` = displacement in direction *i*
        at the receiver due to a unit point force in direction *j*
        at the source.
    """

    src = np.asarray(source_xyz_m, dtype=np.float64)
    recv = np.asarray(receiver_xyz_m, dtype=np.float64)

    offset_xy = recv[:2] - src[:2]
    horizontal_distance_km = float(np.linalg.norm(offset_xy)) / 1000.0
    if horizontal_distance_km <= 0.0:
        raise ValueError("PyFK requires non-zero horizontal source-receiver distance")

    source_depth_km = float(src[2]) / 1000.0
    receiver_depth_km = float(recv[2]) / 1000.0
    azimuth_deg = _station_azimuth_deg(float(offset_xy[0]), float(offset_xy[1]))

    pyfk = _require_pyfk()
    config_class = pyfk["Config"]
    seis_model_class = pyfk["SeisModel"]
    source_model_class = pyfk["SourceModel"]
    utc_date_time_class = pyfk["UTCDateTime"]
    calculate_gf_fn = pyfk["calculate_gf"]
    sync_calculate_gf = pyfk["sync_calculate_gf"]

    model = seis_model_class(LAYER_MODEL, flattening=False)
    pyfk_source = source_model_class(sdep=source_depth_km, srcType="sf")

    config = config_class(
        model=model,
        source=pyfk_source,
        receiver_distance=[horizontal_distance_km],
        npt=int(len(time)),
        dt=float(time[1] - time[0]) if len(time) > 1 else 0.01,
        dk=DK,
        smth=SMTH,
        pmin=PMIN,
        pmax=PMAX,
        kmax=KMAX,
        rdep=receiver_depth_km,
        samples_before_first_arrival=samples_before_first_arrival,
    )
    elementary_gf = calculate_gf_fn(config)

    utc_epoch = utc_date_time_class(1970, 1, 1)
    greens_fn = np.empty((len(time), 3, 3), dtype=np.float64)

    for force_idx, force_dir in enumerate(_FORCE_DIRECTIONS):
        strike, dip = _force_orientation(force_dir)
        pyfk_source.update_source_mechanism([force_amplitude, strike, dip])
        pyfk_source.calculate_radiation_pattern(azimuth_deg)
        stream = sync_calculate_gf(elementary_gf, pyfk_source)[0]

        up = _sync_trace_to_time(stream[0], time, utc_epoch)
        rad = _sync_trace_to_time(stream[1], time, utc_epoch)
        tan = _sync_trace_to_time(stream[2], time, utc_epoch)
        greens_fn[:, :, force_idx] = _urt_to_xyz_down(up, rad, tan, azimuth_deg)

    return greens_fn


def _ricker(time: npt.NDArray[np.float64], freq: float, delay: float) -> npt.NDArray[np.float64]:
    """Ricker wavelet (normalised to peak amplitude 1)."""
    x = np.pi * freq * (time - delay)
    return (1.0 - 2.0 * x**2) * np.exp(-(x**2))


def _match_saved_receiver(
    library_root: Path, receiver_xyz_m: npt.NDArray[np.float64], tolerance_m: float
) -> npt.NDArray[np.float64]:
    """Find the nearest saved SEM source coordinate to *receiver_xyz_m*."""
    index = load_or_rebuild_index(library_root)
    if not index.sources:
        raise ValueError(f"No saved SEM source coordinates found in {library_root}")
    saved = np.asarray([e.source_xyz_m for e in index.sources], dtype=np.float64)
    distances = np.linalg.norm(saved - receiver_xyz_m[None, :], axis=1)
    nearest_idx = int(np.argmin(distances))
    nearest_dist = float(distances[nearest_idx])
    if nearest_dist > tolerance_m:
        raise ValueError(
            f"receiver_xyz {receiver_xyz_m} is {nearest_dist:.3f} m from nearest "
            f"SEM source coordinate {saved[nearest_idx]}; tolerance is {tolerance_m:.3f} m"
        )
    return saved[nearest_idx].copy()


def _read_time_and_stf(
    work_dir: Path,
) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
    """Read the SEM output time grid and STF values from config.h5."""
    with h5py.File(work_dir / "config.h5", "r") as cfg:
        sim = cfg["simulation"].attrs
        stride = int(sim["snapshot_stride"])
        output_dt_s = float(sim["output_dt_s"])
        stf_t = np.asarray(cfg["source/stf_t"], dtype=np.float64)
        stf_v = np.asarray(cfg["source/stf_values"], dtype=np.float64)
    output_time_s = stf_t[::stride]
    source_values = np.interp(output_time_s, stf_t, stf_v)
    return output_time_s, output_dt_s, source_values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PyFK reference Green tensor for a layered half-space."
        "  Convention matches compare.py: --source is the displacement"
        " observation point, --receiver selects the SEM source run."
    )
    p.add_argument(
        "library_root",
        type=Path,
        help="Green function library root (contains source index / greenfun)",
    )
    p.add_argument(
        "--source",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Displacement observation point [m]",
    )
    p.add_argument(
        "--receiver",
        type=float,
        nargs=3,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Point near the SEM source used to select the source run [m]",
    )
    p.add_argument("--output", type=Path, required=True, help="Output .npz path")
    p.add_argument(
        "--source-depth-m",
        type=float,
        default=None,
        help="Override source depth [m]; default uses SEM source depth",
    )
    p.add_argument(
        "--receiver-tolerance-m",
        type=float,
        default=100.0,
        help="Tolerance for matching --receiver to a saved SEM source [m]",
    )
    p.add_argument(
        "--force-amplitude",
        type=float,
        default=FORCE_AMPLITUDE,
        help="PyFK single-force amplitude",
    )
    p.add_argument(
        "--ricker-freq",
        type=float,
        default=None,
        help="If set, convolve with a Ricker wavelet of this frequency [Hz]",
    )
    p.add_argument(
        "--ricker-delay",
        type=float,
        default=1.0,
        help="Ricker wavelet time delay [s] (default 1.0)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    library_root = args.library_root.resolve()
    source_xyz_m = np.asarray(args.source, dtype=np.float64)
    receiver_xyz_m = np.asarray(args.receiver, dtype=np.float64)

    # Match the SEM source run and read the output time grid.
    sem_source_xyz_m = _match_saved_receiver(
        library_root, receiver_xyz_m, args.receiver_tolerance_m
    )
    time, output_dt_s, source_values = _read_time_and_stf(library_root.parent)

    # Place the PyFK source at the matched SEM source location (depth override optional).
    pyfk_source_xyz = sem_source_xyz_m.copy()
    if args.source_depth_m is not None:
        pyfk_source_xyz[2] = max(pyfk_source_xyz[2], args.source_depth_m)

    # Convention: --source = observation point (PyFK receiver),
    #             SEM source = PyFK source (force location).
    # PyFK returns the step response (impulse Green tensor).  The SEM stores
    # displacement convolved with the source time function, so we must apply
    # the same STF to the PyFK result for a fair comparison.
    greens_step = compute_green_tensor(
        source_xyz_m=pyfk_source_xyz,
        receiver_xyz_m=source_xyz_m,
        time=time,
        force_amplitude=args.force_amplitude,
    )

    nt = greens_step.shape[0]
    greens_fn = np.zeros_like(greens_step)
    for i in range(3):
        for j in range(3):
            full = np.convolve(greens_step[:, i, j], source_values, mode="full") * output_dt_s
            greens_fn[:, i, j] = full[:nt]

    output: dict[str, npt.NDArray[np.float64]] = {
        "time": time,
        "source_xyz_m": source_xyz_m,
        "receiver_xyz_m": receiver_xyz_m,
        "sem_source_xyz_m": sem_source_xyz_m,
        "source_time_function": source_values,
        "step_response": greens_step,
        "displacement": greens_fn,
    }

    if args.ricker_freq is not None:
        wavelet = _ricker(time, args.ricker_freq, args.ricker_delay)
        synth = np.zeros_like(greens_fn)
        for i in range(3):
            for j in range(3):
                full = np.convolve(greens_fn[:, i, j], wavelet, mode="full") * output_dt_s
                synth[:, i, j] = full[:nt]
        output["ricker_wavelet"] = wavelet
        output["synthetic_displacement"] = synth
        print(f"Ricker wavelet: f={args.ricker_freq} Hz, delay={args.ricker_delay} s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"Wrote {args.output}")
    print(f"source (obs)     : {source_xyz_m}")
    print(f"receiver (match) : {receiver_xyz_m}")
    print(f"sem_source       : {sem_source_xyz_m}")
    print(f"pyfk_source      : {pyfk_source_xyz}")
    print(f"time             : {time.shape}")
    print(f"displacement     : {greens_fn.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
