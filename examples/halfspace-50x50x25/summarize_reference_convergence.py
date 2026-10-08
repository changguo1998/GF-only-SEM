"""Summarize cached reference convergence, components, and mixed wave windows."""

import csv
import re
from pathlib import Path

import matplotlib
import numpy as np
from check_reference_convergence import metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
CACHE = CASE / "depth_sweep/reference_convergence"
OUTPUT = ROOT / "docs/test-reports/data"
END_TIMES = {2000: 3.5, 5000: 4.5, 8000: 5.5, 11000: 6.5, 14000: 7.3}


def save_csv(name, rows):
    with (OUTPUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


groups = {}
for path in CACHE.glob("depth*_offset*_q*_s*_t*.npz"):
    depth, offset, points, segments, refinement = map(int, re.findall(r"\d+", path.stem))
    groups.setdefault((depth, offset), []).append((refinement, points, segments, path))

convergence_rows, component_rows, window_rows = [], [], []
for (depth, offset), entries in sorted(groups.items(), reverse=True):
    entries.sort()
    with np.load(entries[-1][-1]) as finest:
        finest_reference = finest["analytical"]
    baseline_path = CACHE / f"depth{depth}_offset{offset}_q24_s800_t1.npz"
    with np.load(baseline_path) as baseline:
        baseline_reference = baseline["analytical"]
        baseline_mask = baseline["time"] < END_TIMES[offset]
        baseline_scores = metrics(
            baseline["sem"][baseline_mask], baseline_reference[baseline_mask]
        )
    with (CASE / "depth_sweep" / f"depth_{depth:04d}m" / "metrics.csv").open() as stream:
        published = next(row for row in csv.DictReader(stream) if float(row["offset_m"]) == offset)
    if published.get("reference_dt_s"):
        with np.load(baseline_path) as baseline:
            saved_refinement = round(
                float(baseline["time"][1] - baseline["time"][0])
                / float(published["reference_dt_s"])
            )
        with np.load(
            CACHE / f"depth{depth}_offset{offset}_q48_s1600_t{saved_refinement}.npz"
        ) as saved:
            baseline_scores = metrics(
                saved["sem"][baseline_mask], saved["analytical"][baseline_mask]
            )
    for key, archived_key in (
        ("correlation", "correlation"),
        ("scale", "scale"),
        ("fitted_l2", "fitted_relative_l2"),
    ):
        if not np.isclose(
            baseline_scores[key], float(published[archived_key]), rtol=1e-6, atol=1e-9
        ):
            raise ValueError(f"Baseline mismatch at depth={depth}, offset={offset}: {key}")
    for refinement, points, segments, path in entries:
        with np.load(path) as data:
            time_s = data["time"]
            sem = data["sem"].astype(np.float64)
            analytical = data["analytical"]
        if not np.isfinite(sem).all() or not np.isfinite(analytical).all():
            raise ValueError(f"Nonfinite waveform in {path}")
        valid = time_s < END_TIMES[offset]
        identity = dict(
            source_depth_m=depth,
            offset_m=offset,
            quadrature_points=points,
            segments=segments,
            time_refinement=refinement,
        )
        scores = metrics(sem[valid], analytical[valid])
        convergence_rows.append(
            dict(
                **identity,
                **scores,
                change_from_baseline=np.linalg.norm((analytical - baseline_reference)[valid])
                / np.linalg.norm(baseline_reference[valid]),
                change_from_finest=np.linalg.norm((analytical - finest_reference)[valid])
                / np.linalg.norm(finest_reference[valid]),
            )
        )
        if path not in (entries[-1][-1], baseline_path):
            continue

        # Time gating is heuristic: the S and Rayleigh wave packets overlap.
        split_s = 1.0 + np.hypot(offset, depth) / 3000.0 - 0.6
        windows = {
            "full": valid,
            "early_body_dominated": valid & (time_s < split_s),
            "late_S_R_mixed": valid & (time_s >= split_s),
        }
        residual = sem - analytical
        residual_energy = np.sum(residual[valid] ** 2)
        for label, mask in windows.items():
            window_rows.append(
                dict(
                    **identity,
                    window=label,
                    split_time_s=split_s,
                    **metrics(sem[mask], analytical[mask]),
                    residual_energy_fraction=np.sum(residual[mask] ** 2) / residual_energy,
                )
            )
        for response in range(3):
            for force in range(3):
                component = "xyz"[response] + "xyz"[force]
                component_sem = sem[valid, response, force]
                component_reference = analytical[valid, response, force]
                # Do not report unstable relative metrics for theoretical zeros.
                relative_norm = np.linalg.norm(component_reference) / np.linalg.norm(
                    analytical[valid]
                )
                if relative_norm < 1e-8:
                    component_scores = {
                        key: ""
                        for key in (
                            "correlation",
                            "scale",
                            "fitted_l2",
                            "raw_l2",
                            "sem_norm",
                            "reference_norm",
                        )
                    }
                else:
                    component_scores = metrics(component_sem, component_reference)
                component_rows.append(
                    dict(
                        **identity,
                        component_response_force=component,
                        **component_scores,
                        reference_energy_fraction=relative_norm**2,
                        residual_energy_fraction=np.sum(residual[valid, response, force] ** 2)
                        / residual_energy,
                    )
                )

    print(
        f"depth={depth} offset={offset} finest={entries[-1][:3]} fitted_L2={scores['fitted_l2']:.6f} scale={scores['scale']:.6f}"
    )
    if depth in (1, 10) and offset == 14000:
        figure, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True, layout="constrained")
        for axis, (response, force) in zip(axes, ((0, 0), (0, 2), (2, 0), (2, 2), (1, 1))):
            axis.plot(
                time_s[valid], sem[valid, response, force], label="SEM (unchanged)", linewidth=1.5
            )
            axis.plot(
                time_s[valid],
                baseline_reference[valid, response, force],
                label="Reference, 50 ms",
                linestyle="--",
            )
            axis.plot(
                time_s[valid],
                finest_reference[valid, response, force],
                label=f"Reference, {50 / entries[-1][0]:g} ms",
                linestyle=":",
            )
            axis.axvline(split_s, color="gray", linewidth=0.8)
            axis.set_ylabel("u_" + "xyz"[response] + "xyz"[force] + " (m)")
            axis.grid(alpha=0.3)
        axes[0].legend()
        axes[0].set_title(
            f"{depth} m source depth, 14 km epicentral distance; response-force indices"
        )
        axes[-1].set_xlabel("Time (s)")
        suffix = "-1m" if depth == 1 else ""
        figure.savefig(OUTPUT / f"halfspace-reference-waveforms{suffix}.svg")
        figure.savefig(CACHE / f"waveforms{suffix}.png", dpi=120)
        plt.close(figure)

save_csv("halfspace-reference-convergence.csv", convergence_rows)
save_csv("halfspace-reference-components.csv", component_rows)
save_csv("halfspace-reference-windows.csv", window_rows)
