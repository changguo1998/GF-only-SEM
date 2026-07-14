#!/usr/bin/env python3
"""Plot SEM vs reference Green's function comparison.

Output
------
For each quantity (displacement, velocity, acceleration):
* ``compare_<quantity>_raw.png``       — 3×3 subplot grid, raw amplitudes
* ``compare_<quantity>_normalized.png`` — 3×3 subplot grid, normalized to [-1,1]

Layout (3 rows × 3 columns)
----------------------------
Rows:    force directions  F_x, F_y, F_z
Columns: displacement components  u_x, u_y, u_z

Each subplot has exactly 2 lines:
- Reference (dark blue solid)
- SEM (red solid)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

_FORCE_LABELS = ("F_x", "F_y", "F_z")
_COMP_LABELS = ("u_x", "u_y", "u_z")
_DIRECTIONS = ("x", "y", "z")
_QUANTITIES = ("displacement", "velocity", "acceleration")
_UNITS = {"displacement": "m", "velocity": "m/s", "acceleration": "m/s\u00b2"}

_COLORS = {"reference": "#00008B", "sem": "#d62728"}
_LINESTYLES = {"reference": "-", "sem": "-"}
_LABELS = {"reference": "Reference", "sem": "SEM"}
_LW = {"reference": 1.0, "sem": 0.8}


def _amplitude_annotation(ax, y_pos: float, label: str, amplitude: float, unit: str = "m") -> None:
    if amplitude == 0.0:
        text = f"{label}: 0.0"
    elif abs(amplitude) < 1e-12:
        text = f"{label}: {amplitude:.3e} {unit}"
    elif abs(amplitude) < 1e-9:
        text = f"{label}: {amplitude:.3e} {unit}"
    elif abs(amplitude) < 1.0:
        text = f"{label}: {amplitude:.6f} {unit}"
    else:
        text = f"{label}: {amplitude:.3f} {unit}"
    ax.text(
        0.98,
        y_pos,
        text,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="center",
        horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def _normalize(series: np.ndarray) -> tuple[np.ndarray, float]:
    mx = float(np.max(np.abs(series)))
    if mx <= 0.0:
        return np.zeros_like(series), 0.0
    return series / mx, mx


def _make_comparison_figure(
    time: np.ndarray,
    ref: np.ndarray,
    sem: np.ndarray,
    normalize: bool,
    output_dir: Path,
    suffix: str,
    quantity: str = "displacement",
) -> None:
    """Single figure: 3 rows × 3 columns.

    Rows: force directions  F_x, F_y, F_z
    Columns: displacement components  u_x, u_y, u_z
    """
    q_label = quantity.capitalize()
    unit = _UNITS.get(quantity, "m")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=False)
    title = f"{q_label} Waveforms — Raw" if not normalize else f"{q_label} Waveforms — Normalized"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for fi, fd in enumerate(_DIRECTIONS):  # rows = force direction
        for ci, comp in enumerate(_COMP_LABELS):  # columns = displacement component
            ax = axes[fi, ci]

            ref_series = ref[:, ci, fi]
            sem_series = sem[:, ci, fi]

            if normalize:
                ref_plt, ref_mx = _normalize(ref_series)
                sem_plt, sem_mx = _normalize(sem_series)
            else:
                ref_plt = ref_series
                ref_mx = float(np.max(np.abs(ref_series))) if ref_series.size > 0 else 0.0
                sem_plt = sem_series
                sem_mx = float(np.max(np.abs(sem_series))) if sem_series.size > 0 else 0.0

            # Reference — dark blue solid
            ax.plot(
                time,
                ref_plt,
                color=_COLORS["reference"],
                ls=_LINESTYLES["reference"],
                lw=_LW["reference"],
                label=_LABELS["reference"],
            )
            # SEM — red solid
            ax.plot(
                time,
                sem_plt,
                color=_COLORS["sem"],
                ls=_LINESTYLES["sem"],
                lw=_LW["sem"],
                label=_LABELS["sem"],
            )

            # Annotations
            y_off = 0.0
            if ref_mx > 0:
                _amplitude_annotation(ax, 0.96 - y_off, "Ref", ref_mx, unit)
                y_off += 0.11
            if sem_mx > 0:
                _amplitude_annotation(ax, 0.96 - y_off, "SEM", sem_mx, unit)

            ax.set_title(f"F={fd.upper()}, {comp}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if fi == 2:
                ax.set_xlabel("Time (s)", fontsize=8)
            if ci == 0:
                ax.set_ylabel(
                    q_label if not normalize else f"Normalized {q_label.lower()}", fontsize=8
                )

    # Legend at bottom
    keys = ["reference", "sem"]
    handles = [
        plt.Line2D([0], [0], color=_COLORS[k], ls=_LINESTYLES[k], lw=1.5, label=_LABELS[k])
        for k in keys
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(keys), fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fname = f"compare_{quantity}_{suffix}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir / fname}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot SEM vs reference Green's function comparison."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("lamb_comparison.npz"),
        help="Input .npz (default: ./lamb_comparison.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for PNGs (default: current dir)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Skip plt.show (no effect with Agg)"
    )
    return parser


def _load_quantity(data: np.lib.npyio.NpzFile, prefix: str, time: np.ndarray) -> np.ndarray | None:
    """Load *prefix from data, or compute velocity/acceleration from displacement."""
    ref_key = f"reference_{prefix}"
    sem_key = f"sem_{prefix}"
    if ref_key in data and sem_key in data:
        ref = np.asarray(data[ref_key], dtype=np.float64)
        sem = np.asarray(data[sem_key], dtype=np.float64)
        return ref, sem
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"Loading {args.input} ...")
    data = np.load(args.input, allow_pickle=False)

    time = np.asarray(data["time"], dtype=np.float64)
    print(f"  time: {len(time)} steps [{time[0]:.3f}, {time[-1]:.3f}] s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating plots ...")

    for q in _QUANTITIES:
        pair = _load_quantity(data, q, time)
        if pair is None:
            print(f"  {q}: data not available in .npz, skipping")
            continue
        ref, sem = pair
        print(f"  {q}: ref={ref.shape}, sem={sem.shape}")

        _make_comparison_figure(
            time, ref, sem, normalize=False, output_dir=args.output_dir, suffix="raw", quantity=q
        )
        _make_comparison_figure(
            time,
            ref,
            sem,
            normalize=True,
            output_dir=args.output_dir,
            suffix="normalized",
            quantity=q,
        )

    print("\nAll plots saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
