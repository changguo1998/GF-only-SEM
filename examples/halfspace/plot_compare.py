#!/usr/bin/env python3
"""Plot SEM vs reference Green's function comparison.

Output
------
* ``compare_displacement_raw.png``       — displacement, 3×3 subplot grid, raw amplitudes
* ``compare_displacement_normalized.png`` — displacement, 3×3 subplot grid, normalized to [-1,1]

Layout (3 rows × 3 columns)
----------------------------
Rows:    force directions  F_x, F_y, F_z
Columns: displacement components  u_x, u_y, u_z

Each subplot has exactly 2 lines:
- Reference (dark blue solid)
- SEM (red solid)

*SCALED SEM* (best-fit linear scale) is also plotted if ``amplitude_scale != 0``.
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

_COLORS = {"reference": "#00008B", "sem": "#d62728", "scaled_sem": "#2ca02c"}
_LINESTYLES = {"reference": "-", "sem": "-", "scaled_sem": "-."}
_LABELS = {"reference": "Reference", "sem": "SEM", "scaled_sem": "SEM (scaled)"}
_LW = {"reference": 1.0, "sem": 0.8, "scaled_sem": 0.8}


def _amplitude_annotation(ax, y_pos: float, label: str, amplitude: float) -> None:
    if amplitude == 0.0:
        text = f"{label}: 0.0"
    elif abs(amplitude) < 1e-12:
        text = f"{label}: {amplitude:.3e} m"
    elif abs(amplitude) < 1e-9:
        text = f"{label}: {amplitude:.3e} m"
    elif abs(amplitude) < 1.0:
        text = f"{label}: {amplitude:.6f} m"
    else:
        text = f"{label}: {amplitude:.3f} m"
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
    sem_scaled: np.ndarray | None,
    scale: float,
    normalize: bool,
    output_dir: Path,
    suffix: str,
) -> None:
    """Single figure: 3 rows × 3 columns.

    Rows: force directions  F_x, F_y, F_z
    Columns: displacement components  u_x, u_y, u_z
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=False)
    title = (
        "Displacement Waveforms — Raw" if not normalize else "Displacement Waveforms — Normalized"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for fi, fd in enumerate(_DIRECTIONS):  # rows = force direction
        for ci, comp in enumerate(_COMP_LABELS):  # columns = displacement component
            ax = axes[fi, ci]

            ref_series = ref[:, ci, fi]
            sem_series = sem[:, ci, fi]
            scaled_series = sem_scaled[:, ci, fi] if sem_scaled is not None else None

            if normalize:
                ref_plt, ref_mx = _normalize(ref_series)
                sem_plt, sem_mx = _normalize(sem_series)
                if scaled_series is not None:
                    ssc_plt, ssc_mx = _normalize(scaled_series)
                else:
                    ssc_plt = ssc_mx = None
            else:
                ref_plt = ref_series
                ref_mx = float(np.max(np.abs(ref_series))) if ref_series.size > 0 else 0.0
                sem_plt = sem_series
                sem_mx = float(np.max(np.abs(sem_series))) if sem_series.size > 0 else 0.0
                if scaled_series is not None:
                    ssc_plt = scaled_series
                    ssc_mx = (
                        float(np.max(np.abs(scaled_series))) if scaled_series.size > 0 else 0.0
                    )
                else:
                    ssc_plt = ssc_mx = None

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
            # Scaled SEM if available
            if ssc_plt is not None:
                ax.plot(
                    time,
                    ssc_plt,
                    color=_COLORS["scaled_sem"],
                    ls=_LINESTYLES["scaled_sem"],
                    lw=_LW["scaled_sem"],
                    label=_LABELS["scaled_sem"],
                )

            # Annotations
            y_off = 0.0
            if ref_mx > 0:
                _amplitude_annotation(ax, 0.96 - y_off, "Ref", ref_mx)
                y_off += 0.09
            if sem_mx > 0:
                _amplitude_annotation(ax, 0.96 - y_off, "SEM", sem_mx)
                y_off += 0.09
            if ssc_mx is not None and ssc_mx > 0:
                _amplitude_annotation(ax, 0.96 - y_off, "Scaled", ssc_mx)

            ax.set_title(f"F={fd.upper()}, {comp}", fontsize=9)
            ax.grid(True, alpha=0.25)
            if fi == 2:
                ax.set_xlabel("Time (s)", fontsize=8)
            if ci == 0:
                ax.set_ylabel(
                    "Displacement" if not normalize else "Normalized displacement", fontsize=8
                )

    # Legend at bottom
    keys = ["reference", "sem"]
    if sem_scaled is not None and scale > 0:
        keys.append("scaled_sem")
    handles = [
        plt.Line2D([0], [0], color=_COLORS[k], ls=_LINESTYLES[k], lw=1.5, label=_LABELS[k])
        for k in keys
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(keys), fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fname = f"compare_displacement_{suffix}.png"
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print(f"Loading {args.input} ...")
    data = np.load(args.input, allow_pickle=False)

    time = np.asarray(data["time"], dtype=np.float64)
    ref = np.asarray(data["reference_displacement"], dtype=np.float64)
    sem = np.asarray(data["sem_displacement"], dtype=np.float64)
    sem_scaled = np.asarray(data.get("scaled_sem_displacement", sem), dtype=np.float64)
    scale = float(data.get("amplitude_scale", 0.0))

    print(f"  time: {len(time)} steps [{time[0]:.3f}, {time[-1]:.3f}] s")
    print(f"  ref_displacement:      {ref.shape}")
    print(f"  sem_displacement:      {sem.shape}")
    print(f"  amplitude_scale:       {scale:.6e}")
    print()
    print(f"Note: velocity and acceleration are not stored in the Green's function library.")
    print(f"      Only displacement comparison is available.")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating plots ...")

    scaled_array = sem_scaled if scale > 0 else None

    _make_comparison_figure(
        time,
        ref,
        sem,
        scaled_array,
        scale,
        normalize=False,
        output_dir=args.output_dir,
        suffix="raw",
    )
    _make_comparison_figure(
        time,
        ref,
        sem,
        scaled_array,
        scale,
        normalize=True,
        output_dir=args.output_dir,
        suffix="normalized",
    )

    print("\nAll plots saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
