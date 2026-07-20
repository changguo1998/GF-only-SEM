#!/usr/bin/env python3
"""Multi-point SEM vs PyFK reference comparison for layered model.

Runs entirely in the PyFK venv. Selects multiple observation points,
queries SEM Green's function and PyFK reference for each, and reports
per-point + aggregate statistics.

Usage (from layer example dir):
    .pyfk-venv/bin/python multi_compare.py [--n-points N] [--output multi_comparison.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = (_SCRIPT_DIR / "../..").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from greenfun.library import GreenFunctionLibrary  # noqa: E402

sys.path.insert(0, str(_SCRIPT_DIR))
from reference import compute_green_tensor, _read_time_and_stf  # noqa: E402


def select_source_positions(
    vertex_coords: np.ndarray, sem_source: np.ndarray, n_points: int = 20
) -> tuple[np.ndarray, list[bool]]:
    """Select observation points: half at mesh vertices, half off-grid.

    Excludes points directly above the SEM source (zero horizontal distance
    breaks PyFK).
    """
    from scipy.spatial import KDTree

    tree = KDTree(vertex_coords)
    n_vertex = n_points // 2
    n_offgrid = n_points - n_vertex

    # Vertex points: spread across the domain
    dist_to_source = np.linalg.norm(vertex_coords - sem_source, axis=1)
    horiz_dist = np.linalg.norm(vertex_coords[:, :2] - sem_source[:2], axis=1)
    valid_mask = (dist_to_source > 200.0) & (horiz_dist > 100.0)
    valid_indices = np.where(valid_mask)[0]
    step = max(1, len(valid_indices) // n_vertex)
    vertex_indices = valid_indices[::step][:n_vertex]
    vertex_positions = vertex_coords[vertex_indices]

    # Off-grid points
    coords_min = vertex_coords.min(axis=0)
    coords_max = vertex_coords.max(axis=0)
    rng = np.random.default_rng(42)
    offgrid_positions = []
    for _ in range(n_offgrid):
        for _attempt in range(100):
            pt = rng.uniform(coords_min, coords_max)
            nn_dist, _ = tree.query(pt)
            horiz = np.linalg.norm(pt[:2] - sem_source[:2])
            if nn_dist > 10.0 and horiz > 100.0:
                offgrid_positions.append(pt)
                break

    positions = np.vstack([vertex_positions, np.array(offgrid_positions)])
    is_vertex = [True] * len(vertex_positions) + [False] * len(offgrid_positions)
    return positions, is_vertex


def compute_rel_l2(sem: np.ndarray, ref: np.ndarray) -> float:
    errors = []
    for i in range(3):
        s = sem[:, i, i]
        r = ref[:, i, i]
        norm_r = np.linalg.norm(r)
        if norm_r > 1e-30:
            errors.append(np.linalg.norm(s - r) / norm_r)
    return float(np.mean(errors)) if errors else float("nan")


def compute_correlation(sem: np.ndarray, ref: np.ndarray) -> float:
    corrs = []
    for i in range(3):
        s = sem[:, i, i]
        r = ref[:, i, i]
        if np.std(s) > 1e-30 and np.std(r) > 1e-30:
            corrs.append(np.corrcoef(s, r)[0, 1])
    return float(np.mean(corrs)) if corrs else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-point SEM vs PyFK comparison")
    parser.add_argument("--library", default="greenfun", help="Green's function library root")
    parser.add_argument("--n-points", type=int, default=20, help="Number of source positions")
    parser.add_argument("--output", default="multi_comparison.npz", help="Output NPZ file")
    parser.add_argument("--source-depth-m", type=float, default=250.0, help="PyFK source depth")
    args = parser.parse_args(argv)

    print("=== Loading Green's function library ===")
    lib = GreenFunctionLibrary(args.library)
    print(f"  Sources: {lib.n_sources}, Tiles: {lib.n_tiles}")

    sem_source = lib._sources_by_index[0].source_xyz_m
    print(f"  SEM source (receiver): {sem_source}")

    source_run = lib._get_or_create_source_run(lib._sources_by_index[0])
    source_run.load()
    vertex_coords = source_run.vertex_coords
    print(f"  Recorded GLL nodes: {len(vertex_coords)}")

    # Read time grid and STF from the example directory
    time, output_dt_s, source_values = _read_time_and_stf(Path(args.library).parent)
    print(f"  Time: {len(time)} steps, dt={output_dt_s}s")

    # PyFK source position (force location)
    pyfk_source = sem_source.copy()
    pyfk_source[2] = max(pyfk_source[2], args.source_depth_m)

    positions, is_vertex = select_source_positions(vertex_coords, sem_source, args.n_points)
    print(
        f"\n=== Selected {len(positions)} positions ({sum(is_vertex)} vertex, {sum(not v for v in is_vertex)} off-grid) ==="
    )

    results = []
    for idx, (pos, is_vtx) in enumerate(zip(positions, is_vertex)):
        label = "vertex" if is_vtx else "offgrid"
        print(f"\n--- Point {idx + 1}/{len(positions)} [{label}] at {pos} ---")

        # SEM query
        try:
            sem_result = lib.query(pos, sem_source, quantity="displacement")
            sem_disp = sem_result.displacement
            if sem_disp is None:
                print(f"  SEM: no displacement, skipping")
                continue
            interp = sem_result.interpolation_used
            print(f"  SEM: interpolated={interp}, max_abs={np.max(np.abs(sem_disp)):.4e}")
        except Exception as e:
            print(f"  SEM error: {e}")
            continue

        # PyFK reference
        try:
            greens_step = compute_green_tensor(
                source_xyz_m=pyfk_source, receiver_xyz_m=pos.astype(np.float64), time=time
            )
            nt = greens_step.shape[0]
            ref_disp = np.zeros_like(greens_step)
            for i in range(3):
                for j in range(3):
                    full = (
                        np.convolve(greens_step[:, i, j], source_values, mode="full") * output_dt_s
                    )
                    ref_disp[:, i, j] = full[:nt]
            print(f"  Ref: max_abs={np.max(np.abs(ref_disp)):.4e}")
        except Exception as e:
            print(f"  PyFK error: {e}")
            continue

        # Best-fit scale
        sem_flat = sem_disp.ravel()
        ref_flat = ref_disp.ravel()
        scale = float(np.dot(ref_flat, sem_flat) / (np.dot(sem_flat, sem_flat) + 1e-30))
        scaled_sem = sem_disp * scale

        rel_l2 = compute_rel_l2(scaled_sem, ref_disp)
        corr = compute_correlation(scaled_sem, ref_disp)
        print(f"  scale={scale:.4e}, rel_l2={rel_l2:.4f}, corr={corr:.4f}")

        results.append(
            {
                "idx": idx,
                "position": pos,
                "is_vertex": is_vtx,
                "interpolated": interp,
                "scale": scale,
                "rel_l2": rel_l2,
                "correlation": corr,
            }
        )

    # Aggregate
    print(f"\n{'=' * 60}")
    print(f"=== Aggregate Results ({len(results)} points) ===")
    print(f"{'=' * 60}")

    for label, group in [
        ("Vertex", [r for r in results if r["is_vertex"]]),
        ("Off-grid", [r for r in results if not r["is_vertex"]]),
        ("All", results),
    ]:
        if not group:
            continue
        rel_l2s = [r["rel_l2"] for r in group]
        corrs = [r["correlation"] for r in group]
        print(f"\n  {label} ({len(group)} points):")
        print(
            f"    rel_l2:  mean={np.mean(rel_l2s):.4f}, median={np.median(rel_l2s):.4f}, min={np.min(rel_l2s):.4f}, max={np.max(rel_l2s):.4f}"
        )
        print(
            f"    corr:    mean={np.mean(corrs):.4f}, median={np.median(corrs):.4f}, min={np.min(corrs):.4f}, max={np.max(corrs):.4f}"
        )

    np.savez(
        args.output,
        positions=np.array([r["position"] for r in results]),
        is_vertex=np.array([r["is_vertex"] for r in results]),
        interpolated=np.array([r["interpolated"] for r in results]),
        scales=np.array([r["scale"] for r in results]),
        rel_l2=np.array([r["rel_l2"] for r in results]),
        correlation=np.array([r["correlation"] for r in results]),
    )
    print(f"\nSaved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
