#!/usr/bin/env python3
"""Plot SEM vs reference Green's function comparison.

Produces 2 figures:

* ``compare_raw.png``       — raw (unnormalized) waveforms, 18 subplots (6×3)
* ``compare_normalized.png`` — normalized waveforms, 18 subplots (6×3)

Layout (both figures)
---------------------
Columns:  force directions  F_x, F_y, F_z
Rows 1-3: displacement components  u_x, u_y, u_z
Rows 4-6: difference (SEM − Reference) for each component

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

_COLORS = {
    "reference": "#00008B",  # dark blue solid
    "sem": "#d62728",  # red solid
    "scaled_sem": "#2ca02c",  # green dash-dot
    "difference": "#00008B",  # dark blue dashed for diff
}
_LINESTYLES = {"reference": "-", "sem": "-", "scaled_sem": "-.", "difference": "--"}
_LABELS = {
    "reference": "Reference",
    "sem": "SEM",
    "scaled_sem": "SEM (scaled)",
    "difference": "Difference",
}
_LW = {"reference": 1.0, "sem": 0.8, "scaled_sem": 0.8, "difference": 0.6}


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
        fontsize=6.5,
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


def _plot_component(
    ax,
    time: np.ndarray,
    ref_series: np.ndarray,
    sem_series: np.ndarray,
    sem_scaled_series: np.ndarray | None,
    normalize: bool,
) -> None:
    if normalize:
        ref_plt, ref_mx = _normalize(ref_series)
        sem_plt, sem_mx = _normalize(sem_series)
        if sem_scaled_series is not None:
            ssc_plt, ssc_mx = _normalize(sem_scaled_series)
        else:
            ssc_plt = ssc_mx = None
    else:
        ref_plt = ref_series
        ref_mx = float(np.max(np.abs(ref_series))) if ref_series.size > 0 else 0.0
        sem_plt = sem_series
        sem_mx = float(np.max(np.abs(sem_series))) if sem_series.size > 0 else 0.0
        if sem_scaled_series is not None:
            ssc_plt = sem_scaled_series
            ssc_mx = (
                float(np.max(np.abs(sem_scaled_series))) if sem_scaled_series.size > 0 else 0.0
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
        y_off += 0.08
    if sem_mx > 0:
        _amplitude_annotation(ax, 0.96 - y_off, "SEM", sem_mx)
        y_off += 0.08
    if ssc_mx is not None and ssc_mx > 0:
        _amplitude_annotation(ax, 0.96 - y_off, "Scaled", ssc_mx)

    ax.grid(True, alpha=0.25)


def _plot_difference(
    ax, time: np.ndarray, ref_series: np.ndarray, sem_series: np.ndarray, normalize: bool
) -> None:
    diff = sem_series - ref_series
    if normalize:
        diff_plt, diff_mx = _normalize(diff)
    else:
        diff_plt = diff
        diff_mx = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0

    ax.plot(
        time,
        diff_plt,
        color=_COLORS["difference"],
        ls=_LINESTYLES["difference"],
        lw=_LW["difference"],
        label=_LABELS["difference"],
    )
    ax.axhline(0, color="gray", lw=0.4)
    if diff_mx > 0:
        _amplitude_annotation(ax, 0.96, "Max diff", diff_mx)
    ax.grid(True, alpha=0.25)


def _make_comparison_figure(
    time: np.ndarray,
    ref: np.ndarray,
    sem: np.ndarray,
    sem_scaled: np.ndarray | None,
    scale: float,
    normalize: bool,
    output_dir: Path,
    suffix: str,
    title_suffix: str,
) -> None:
    """Single figure: 6 rows × 3 columns.

    Rows 1-3: displacement components  u_x, u_y, u_z
    Rows 4-6: difference SEM − Reference for each component
    """
    fig, axes = plt.subplots(6, 3, figsize=(16, 18), sharex=True, sharey=False)
    fig.suptitle(f"Waveform Comparison — {title_suffix}", fontsize=14, fontweight="bold")

    for fi, fd in enumerate(_DIRECTIONS):  # columns = force directions
        for ci, comp in enumerate(_COMP_LABELS):  # rows 1-3 = displacement components
            # ── Row 1-3: displacement comparison ──
            ax = axes[ci, fi]
            ref_series = ref[:, ci, fi]
            sem_series = sem[:, ci, fi]
            scaled_series = sem_scaled[:, ci, fi] if sem_scaled is not None else None
            _plot_component(ax, time, ref_series, sem_series, scaled_series, normalize)

            ax.set_title(f"F={fd.upper()}, {comp}", fontsize=9)
            if ci == 2:
                ax.set_xlabel("Time (s)", fontsize=8)
            if fi == 0:
                ax.set_ylabel(
                    "Displacement" if not normalize else "Normalized displacement", fontsize=8
                )

            # ── Row 4-6: difference ──
            ax = axes[ci + 3, fi]
            _plot_difference(ax, time, ref_series, sem_series, normalize)
            ax.set_title(f"F={fd.upper()}, {comp} — diff", fontsize=9)
            if ci + 3 == 5:
                ax.set_xlabel("Time (s)", fontsize=8)
            if fi == 0:
                ax.set_ylabel(
                    "Difference" if not normalize else "Normalized difference", fontsize=8
                )

    # Legend
    keys = ["reference", "sem"]
    if sem_scaled is not None and scale > 0:
        keys.append("scaled_sem")
    handles = [
        plt.Line2D([0], [0], color=_COLORS[k], ls=_LINESTYLES[k], lw=1.5, label=_LABELS[k])
        for k in keys
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(keys), fontsize=10)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fname = f"compare_{suffix}.png"
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

    # Figure 1 — raw (unnormalized)
    _make_comparison_figure(
        time,
        ref_displacement,
        sem_displacement,
        sem_scaled if scale > 0 else None,
        scale,
        normalize=False,
        output_dir=args.output_dir,
        suffix="raw",
        title_suffix="Raw Waveforms",
    )

    # Figure 2 — normalized
    _make_comparison_figure(
        time,
        ref_displacement,
        sem_displacement,
        sem_scaled if scale > 0 else None,
        scale,
        normalize=True,
        output_dir=args.output_dir,
        suffix="normalized",
        title_suffix="Normalized Waveforms",
    )

    print("\nAll plots saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
