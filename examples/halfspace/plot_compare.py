#!/usr/bin/env python3
"""Plot comparison between Lamb reference and SEM Green's function results.

All time series are normalized for visual comparison; text annotations
report the actual (unnormalized) amplitude values.

Usage:
    python plot_compare.py                          # uses ./lamb_comparison.npz
    python plot_compare.py --input path/to/file.npz
    python plot_compare.py --no-show                # save only, no display
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # headless-safe

_FORCE_LABELS = ("F_x", "F_y", "F_z")
_COMP_LABELS = ("u_x", "u_y", "u_z")
_DIRECTIONS = ("x", "y", "z")
_COLORS = {
    "reference": "#1f77b4",
    "sem": "#d62728",
    "scaled_sem": "#2ca02c",
    "difference": "#9467bd",
}
_LINESTYLES = {"reference": "-", "sem": "--", "scaled_sem": ":", "difference": "-."}
_LABELS = {
    "reference": "Reference (Lamb)",
    "sem": "SEM (raw)",
    "scaled_sem": "SEM (best-fit scale)",
    "difference": "Difference",
}


def _amplitude_annotation(ax, y_pos: float, label: str, amplitude: float) -> None:
    """Add a text annotation showing the actual amplitude next to the plot."""
    if amplitude == 0.0:
        text = f"{label}: 0.0"
    elif amplitude < 1e-6:
        text = f"{label}: {amplitude:.3e} m"
    elif amplitude < 1.0:
        text = f"{label}: {amplitude:.3f} m"
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
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
    )


def _normalize(series: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize to [-1, 1]; return (normalized, max_abs)."""
    mx = float(np.max(np.abs(series)))
    if mx <= 0.0:
        return np.zeros_like(series), 0.0
    return series / mx, mx


def _make_waveform_plot(
    time: np.ndarray,
    ref: np.ndarray,
    sem: np.ndarray,
    sem_scaled: np.ndarray,
    scale: float,
    output_dir: Path,
) -> None:
    """3x3 grid: rows=force direction, cols=displacement component."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=False)
    fig.suptitle("Waveform Comparison — Normalized", fontsize=14, fontweight="bold")

    for fi, fd in enumerate(_DIRECTIONS):
        for ci, comp in enumerate(_COMP_LABELS):
            ax = axes[fi, ci]

            ref_series = ref[:, ci, fi]
            sem_series = sem[:, ci, fi]
            sem_scaled_series = sem_scaled[:, ci, fi]

            # Normalize each series independently
            ref_norm, ref_max = _normalize(ref_series)
            sem_norm, sem_max = _normalize(sem_series)
            ssc_norm, ssc_max = _normalize(sem_scaled_series)

            ax.plot(
                time,
                ref_norm,
                color=_COLORS["reference"],
                ls=_LINESTYLES["reference"],
                lw=1.2,
                label=_LABELS["reference"],
            )
            ax.plot(
                time,
                sem_norm,
                color=_COLORS["sem"],
                ls=_LINESTYLES["sem"],
                lw=1.0,
                alpha=0.7,
                label=_LABELS["sem"],
            )
            if scale > 0:
                ax.plot(
                    time,
                    ssc_norm,
                    color=_COLORS["scaled_sem"],
                    ls=_LINESTYLES["scaled_sem"],
                    lw=1.0,
                    alpha=0.8,
                    label=_LABELS["scaled_sem"],
                )

            # Annotate actual amplitudes
            _amplitude_annotation(ax, 0.96, "Ref max", ref_max)
            _amplitude_annotation(ax, 0.88, "SEM max", sem_max)
            if scale > 0:
                _amplitude_annotation(ax, 0.80, "Scaled max", ssc_max)

            ax.set_title(f"F={fd.upper()}, comp={comp}")
            ax.grid(True, alpha=0.3)
            if fi == 2:
                ax.set_xlabel("Time (s)")
            if ci == 0:
                ax.set_ylabel("Normalized amplitude")

    # Single legend at bottom
    handles = [
        plt.Line2D([0], [0], color=_COLORS[k], ls=_LINESTYLES[k], lw=1.5, label=_LABELS[k])
        for k in ("reference", "sem", "scaled_sem")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_dir / "compare_waveforms.png", dpi=150)
    print(f"  Saved {output_dir / 'compare_waveforms.png'}")
    plt.close(fig)


def _make_direction_plots(
    time: np.ndarray,
    ref: np.ndarray,
    sem: np.ndarray,
    sem_scaled: np.ndarray,
    scale: float,
    output_dir: Path,
) -> None:
    """One panel per force direction, 3 subplots (one per component)."""
    for fi, fd in enumerate(_DIRECTIONS):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f"Direction {fd.upper()} — Normalized, best-fit scale={scale:.3e}",
            fontsize=13,
            fontweight="bold",
        )

        for ci, comp in enumerate(_COMP_LABELS):
            ax = axes[ci]

            ref_series = ref[:, ci, fi]
            sem_series = sem[:, ci, fi]
            sem_scaled_series = sem_scaled[:, ci, fi]

            ref_norm, ref_max = _normalize(ref_series)
            sem_norm, sem_max = _normalize(sem_series)
            ssc_norm, ssc_max = _normalize(sem_scaled_series)

            ax.plot(
                time,
                ref_norm,
                color=_COLORS["reference"],
                ls=_LINESTYLES["reference"],
                lw=1.2,
                label=_LABELS["reference"],
            )
            ax.plot(
                time,
                sem_norm,
                color=_COLORS["sem"],
                ls=_LINESTYLES["sem"],
                lw=1.0,
                alpha=0.7,
                label=_LABELS["sem"],
            )
            if scale > 0:
                ax.plot(
                    time,
                    ssc_norm,
                    color=_COLORS["scaled_sem"],
                    ls=_LINESTYLES["scaled_sem"],
                    lw=1.0,
                    alpha=0.8,
                    label=_LABELS["scaled_sem"],
                )

            _amplitude_annotation(ax, 0.96, "Ref max", ref_max)
            _amplitude_annotation(ax, 0.88, "SEM max", sem_max)
            if scale > 0:
                _amplitude_annotation(ax, 0.80, "Scaled max", ssc_max)

            ax.set_title(f"Component {comp}")
            ax.grid(True, alpha=0.3)
            if ci == 2:
                ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized amplitude")

        handles = [
            plt.Line2D([0], [0], color=_COLORS[k], ls=_LINESTYLES[k], lw=1.5, label=_LABELS[k])
            for k in ("reference", "sem", "scaled_sem")
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        path = output_dir / f"compare_direction_{fd}.png"
        fig.savefig(path, dpi=150)
        print(f"  Saved {path}")
        plt.close(fig)


def _make_difference_plot(
    time: np.ndarray,
    ref: np.ndarray,
    sem: np.ndarray,
    sem_scaled: np.ndarray,
    scale: float,
    output_dir: Path,
) -> None:
    """Difference (error) plot — 3x3 grid showing normalized difference + amplitude."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=False)
    fig.suptitle("Difference (SEM − Reference) — Normalized", fontsize=14, fontweight="bold")

    for fi, fd in enumerate(_DIRECTIONS):
        for ci, comp in enumerate(_COMP_LABELS):
            ax = axes[fi, ci]

            ref_series = ref[:, ci, fi]
            sem_series = sem[:, ci, fi]
            diff_raw = sem_series - ref_series
            sem_scaled_series = sem_scaled[:, ci, fi] if scale > 0 else sem_series
            diff_scaled = sem_scaled_series - ref_series

            diff_raw_norm, diff_raw_max = _normalize(diff_raw)
            diff_scaled_norm, diff_scaled_max = _normalize(diff_scaled)

            ax.plot(
                time,
                diff_raw_norm,
                color=_COLORS["sem"],
                ls=_LINESTYLES["sem"],
                lw=1.0,
                alpha=0.7,
                label="Raw difference",
            )
            if scale > 0:
                ax.plot(
                    time,
                    diff_scaled_norm,
                    color=_COLORS["scaled_sem"],
                    ls=_LINESTYLES["scaled_sem"],
                    lw=1.0,
                    alpha=0.8,
                    label="Scaled difference",
                )

            _amplitude_annotation(ax, 0.96, "Raw diff max", diff_raw_max)
            if scale > 0:
                _amplitude_annotation(ax, 0.88, "Scaled diff max", diff_scaled_max)

            ax.set_title(f"F={fd.upper()}, comp={comp}")
            ax.grid(True, alpha=0.3)
            if fi == 2:
                ax.set_xlabel("Time (s)")
            if ci == 0:
                ax.set_ylabel("Normalized amplitude")

    handles = [
        plt.Line2D([0], [0], color=_COLORS[k], ls=_LINESTYLES[k], lw=1.5, label=label)
        for k, label in [("sem", "Raw difference"), ("scaled_sem", "Scaled difference")]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_dir / "compare_difference.png", dpi=150)
    print(f"  Saved {output_dir / 'compare_difference.png'}")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Lamb vs SEM comparison.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("lamb_comparison.npz"),
        help="Input .npz from compare.py (default: ./lamb_comparison.npz)",
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
    ref_displacement = np.asarray(data["reference_displacement"], dtype=np.float64)
    sem_displacement = np.asarray(data["sem_displacement"], dtype=np.float64)
    sem_scaled = np.asarray(
        data.get("scaled_sem_displacement", sem_displacement), dtype=np.float64
    )
    scale = float(data.get("amplitude_scale", 0.0))

    print(f"  time: {len(time)} steps [{time[0]:.3f}, {time[-1]:.3f}] s")
    print(f"  ref_displacement:      {ref_displacement.shape}")
    print(f"  sem_displacement:      {sem_displacement.shape}")
    print(f"  amplitude_scale:       {scale:.6e}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots ...")

    # ── 3×3 grid ──
    _make_waveform_plot(
        time, ref_displacement, sem_displacement, sem_scaled, scale, args.output_dir
    )

    # ── Per-direction panels ──
    _make_direction_plots(
        time, ref_displacement, sem_displacement, sem_scaled, scale, args.output_dir
    )

    # ── Difference plot ──
    _make_difference_plot(
        time, ref_displacement, sem_displacement, sem_scaled, scale, args.output_dir
    )

    print("\nAll plots saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
