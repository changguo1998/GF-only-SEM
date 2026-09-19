#!/usr/bin/env python3
"""Multi-point SEM vs analytic reference comparison.

Selects deterministic surface positions at several offsets and azimuths,
queries SEM Green's function and analytic Lamb reference for each, and reports
per-point + aggregate statistics for both the full trace and the main-wave
window.

Usage (from halfspace example dir):
    python multi_compare.py [--n-points N] [--output multi_comparison.npz]
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

# Import the Lamb analytic reference
sys.path.insert(0, str(_SCRIPT_DIR))
from reference import compute_reference_result  # noqa: E402


def select_source_positions(
    vertex_coords: np.ndarray, sem_source: np.ndarray, n_points: int = 10
) -> tuple[np.ndarray, list[bool]]:
    """Select fixed surface positions at several offsets from the SEM source.

    Returns (positions, is_vertex_flags).
    """
    from scipy.spatial import KDTree

    tree = KDTree(vertex_coords)
    diagonal_offset_m = 1000.0 / np.sqrt(2.0)
    horizontal_offsets_m = np.array(
        [
            [500.0, 0.0],
            [1000.0, 0.0],
            [1500.0, 0.0],
            [-500.0, 0.0],
            [-1000.0, 0.0],
            [-1500.0, 0.0],
            [0.0, 1000.0],
            [0.0, -1000.0],
            [diagonal_offset_m, diagonal_offset_m],
            [-diagonal_offset_m, -diagonal_offset_m],
        ],
        dtype=np.float64,
    )
    if not 1 <= n_points <= len(horizontal_offsets_m):
        raise ValueError(f"n_points must be between 1 and {len(horizontal_offsets_m)}")

    positions = np.column_stack(
        [
            sem_source[0] + horizontal_offsets_m[:n_points, 0],
            sem_source[1] + horizontal_offsets_m[:n_points, 1],
            np.zeros(n_points, dtype=np.float64),
        ]
    )
    coords_min = vertex_coords.min(axis=0)
    coords_max = vertex_coords.max(axis=0)
    if np.any(positions < coords_min) or np.any(positions > coords_max):
        raise ValueError("selected surface comparison point lies outside recorded bounds")

    nearest_distances, _ = tree.query(positions)
    is_vertex = list(nearest_distances < 1.0e-6)
    return positions, is_vertex


def compute_rel_l2(sem: np.ndarray, ref: np.ndarray) -> float:
    """Compute full-tensor relative L2 error."""
    norm_ref = float(np.linalg.norm(ref.ravel()))
    if norm_ref <= 1.0e-30:
        return float("nan")
    return float(np.linalg.norm((sem - ref).ravel()) / norm_ref)


def compute_correlation(sem: np.ndarray, ref: np.ndarray) -> float:
    """Compute full-tensor Pearson correlation."""
    sem_flat = sem.ravel()
    ref_flat = ref.ravel()
    if np.std(sem_flat) <= 1.0e-30 or np.std(ref_flat) <= 1.0e-30:
        return float("nan")
    return float(np.corrcoef(sem_flat, ref_flat)[0, 1])


def compute_best_fit(sem: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """Return SEM scale, fitted relative L2, and correlation."""
    sem_flat = sem.ravel()
    ref_flat = ref.ravel()
    scale = float(np.dot(ref_flat, sem_flat) / (np.dot(sem_flat, sem_flat) + 1.0e-30))
    scaled_sem = sem * scale
    return scale, compute_rel_l2(scaled_sem, ref), compute_correlation(scaled_sem, ref)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-point SEM vs reference comparison")
    parser.add_argument("--library", default="greenfun", help="Green's function library root")
    parser.add_argument("--n-points", type=int, default=10, help="Number of surface positions")
    parser.add_argument("--output", default="multi_comparison.npz", help="Output NPZ file")
    parser.add_argument(
        "--source-depth-m", type=float, default=278.0, help="Analytic source depth"
    )
    parser.add_argument(
        "--early-end-s", type=float, default=2.0, help="End time of main-wave window"
    )
    args = parser.parse_args(argv)

    print("=== Loading Green's function library ===")
    lib = GreenFunctionLibrary(args.library)
    print(f"  Sources: {lib.n_sources}, Tiles: {lib.n_tiles}")

    # Get the SEM source (receiver) position
    sem_source = lib._sources_by_index[0].source_xyz_m
    print(f"  SEM source (receiver): {sem_source}")

    # Load the SourceRun to get vertex coordinates
    source_run = lib._get_or_create_source_run(lib._sources_by_index[0])
    source_run.load()
    vertex_coords = source_run.vertex_coords
    print(f"  Recorded GLL nodes: {len(vertex_coords)}")

    # Select source positions
    positions, is_vertex = select_source_positions(vertex_coords, sem_source, args.n_points)
    print(
        f"\n=== Selected {len(positions)} source positions ({sum(is_vertex)} vertex, {sum(not v for v in is_vertex)} off-grid) ==="
    )

    # Run comparisons
    results = []
    for idx, (pos, is_vtx) in enumerate(zip(positions, is_vertex)):
        label = "vertex" if is_vtx else "offgrid"
        print(f"\n--- Point {idx + 1}/{len(positions)} [{label}] at {pos} ---")

        # SEM query (displacement)
        try:
            sem_result = lib.query(pos, sem_source, quantity="displacement")
            sem_disp = sem_result.displacement
            if sem_disp is None:
                print(f"  SEM: no displacement data, skipping")
                continue
            interp = sem_result.interpolation_used
            print(f"  SEM: interpolated={interp}, max_abs={np.max(np.abs(sem_disp)):.4e}")
        except Exception as e:
            print(f"  SEM error: {e}")
            continue

        # Reference (Lamb analytic)
        try:
            ref_result = compute_reference_result(
                library_root=Path(args.library),
                source_xyz_m=pos.astype(np.float64),
                receiver_xyz_m=sem_source.astype(np.float64),
                quantity="displacement",
                receiver_tolerance_m=1000.0,
                model_relative_tolerance=0.1,
                source_depth_m=args.source_depth_m,
                n_quad_pts=32,
                n_seg=12,
            )
            ref_disp = ref_result["displacement"]
            print(f"  Ref: max_abs={np.max(np.abs(ref_disp)):.4e}")
        except Exception as e:
            print(f"  Reference error: {e}")
            continue

        # Full-trace and main-wave metrics
        scale, rel_l2, corr = compute_best_fit(sem_disp, ref_disp)
        early_mask = source_run.time < args.early_end_s
        early_scale, early_rel_l2, early_corr = compute_best_fit(
            sem_disp[early_mask], ref_disp[early_mask]
        )
        horizontal_distance_m = float(np.linalg.norm(pos[:2] - sem_source[:2]))
        print(
            f"  distance={horizontal_distance_m:.0f}m, full: scale={scale:.4f}, "
            f"rel_l2={rel_l2:.4f}, corr={corr:.4f}"
        )
        print(
            f"  0-{args.early_end_s:g}s: scale={early_scale:.4f}, "
            f"rel_l2={early_rel_l2:.4f}, corr={early_corr:.4f}"
        )

        results.append(
            {
                "idx": idx,
                "position": pos,
                "is_vertex": is_vtx,
                "interpolated": interp,
                "scale": scale,
                "rel_l2": rel_l2,
                "correlation": corr,
                "early_scale": early_scale,
                "early_rel_l2": early_rel_l2,
                "early_correlation": early_corr,
                "sem_disp": sem_disp,
                "ref_disp": ref_disp,
            }
        )

    # Aggregate
    print(f"\n{'=' * 60}")
    print(f"=== Aggregate Results ({len(results)} points) ===")
    print(f"{'=' * 60}")

    vertex_results = [r for r in results if r["is_vertex"]]
    offgrid_results = [r for r in results if not r["is_vertex"]]

    for label, group in [
        ("Vertex", vertex_results),
        ("Off-grid", offgrid_results),
        ("All", results),
    ]:
        if not group:
            continue
        rel_l2s = [r["rel_l2"] for r in group]
        corrs = [r["correlation"] for r in group]
        early_rel_l2s = [r["early_rel_l2"] for r in group]
        early_corrs = [r["early_correlation"] for r in group]
        print(f"\n  {label} ({len(group)} points):")
        print(
            f"    rel_l2:  mean={np.mean(rel_l2s):.4f}, median={np.median(rel_l2s):.4f}, min={np.min(rel_l2s):.4f}, max={np.max(rel_l2s):.4f}"
        )
        print(
            f"    corr:    mean={np.mean(corrs):.4f}, median={np.median(corrs):.4f}, min={np.min(corrs):.4f}, max={np.max(corrs):.4f}"
        )
        print(
            f"    early rel_l2: mean={np.mean(early_rel_l2s):.4f}, "
            f"min={np.min(early_rel_l2s):.4f}, max={np.max(early_rel_l2s):.4f}"
        )
        print(
            f"    early corr:   mean={np.mean(early_corrs):.4f}, "
            f"min={np.min(early_corrs):.4f}, max={np.max(early_corrs):.4f}"
        )

    # Save results
    save_dict = {
        "positions": np.array([r["position"] for r in results]),
        "is_vertex": np.array([r["is_vertex"] for r in results]),
        "interpolated": np.array([r["interpolated"] for r in results]),
        "scales": np.array([r["scale"] for r in results]),
        "rel_l2": np.array([r["rel_l2"] for r in results]),
        "correlation": np.array([r["correlation"] for r in results]),
        "early_scales": np.array([r["early_scale"] for r in results]),
        "early_rel_l2": np.array([r["early_rel_l2"] for r in results]),
        "early_correlation": np.array([r["early_correlation"] for r in results]),
    }
    np.savez(args.output, **save_dict)
    print(f"\nSaved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
