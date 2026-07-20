#!/usr/bin/env python3
"""Multi-point SEM vs analytic reference comparison.

Selects multiple source positions (mix of mesh vertices and off-grid points),
queries SEM Green's function and analytic Lamb reference for each, and reports
per-point + aggregate statistics.

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
    vertex_coords: np.ndarray, sem_source: np.ndarray, n_points: int = 20
) -> tuple[np.ndarray, list[bool]]:
    """Select source positions: half at mesh vertices, half off-grid.

    Returns (positions, is_vertex_flags).
    """
    from scipy.spatial import KDTree

    tree = KDTree(vertex_coords)
    n_vertex = n_points // 2
    n_offgrid = n_points - n_vertex

    # Vertex points: spread across the domain, exclude points too close to SEM source
    dist_to_source = np.linalg.norm(vertex_coords - sem_source, axis=1)
    # Sort by distance from source, pick spread-out points
    valid_mask = dist_to_source > 200.0  # at least 200m from SEM source
    valid_indices = np.where(valid_mask)[0]
    # Subsample evenly
    step = max(1, len(valid_indices) // n_vertex)
    vertex_indices = valid_indices[::step][:n_vertex]

    vertex_positions = vertex_coords[vertex_indices]

    # Off-grid points: random positions within the domain, not at vertices
    coords_min = vertex_coords.min(axis=0)
    coords_max = vertex_coords.max(axis=0)
    rng = np.random.default_rng(42)
    offgrid_positions = []
    for _ in range(n_offgrid):
        for _attempt in range(100):
            pt = rng.uniform(coords_min, coords_max)
            # Ensure not too close to any vertex (at least 10m away)
            nn_dist, _ = tree.query(pt)
            if nn_dist > 10.0:
                # Also ensure within recording cell bounds
                offgrid_positions.append(pt)
                break

    positions = np.vstack([vertex_positions, np.array(offgrid_positions)])
    is_vertex = [True] * len(vertex_positions) + [False] * len(offgrid_positions)
    return positions, is_vertex


def compute_rel_l2(sem: np.ndarray, ref: np.ndarray) -> float:
    """Compute relative L2 error for diagonal displacement components."""
    errors = []
    for i in range(3):
        s = sem[:, i, i]
        r = ref[:, i, i]
        norm_r = np.linalg.norm(r)
        if norm_r > 1e-30:
            errors.append(np.linalg.norm(s - r) / norm_r)
    return float(np.mean(errors)) if errors else float("nan")


def compute_correlation(sem: np.ndarray, ref: np.ndarray) -> float:
    """Compute mean correlation for diagonal displacement components."""
    corrs = []
    for i in range(3):
        s = sem[:, i, i]
        r = ref[:, i, i]
        if np.std(s) > 1e-30 and np.std(r) > 1e-30:
            corrs.append(np.corrcoef(s, r)[0, 1])
    return float(np.mean(corrs)) if corrs else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-point SEM vs reference comparison")
    parser.add_argument("--library", default="greenfun", help="Green's function library root")
    parser.add_argument("--n-points", type=int, default=20, help="Number of source positions")
    parser.add_argument("--output", default="multi_comparison.npz", help="Output NPZ file")
    parser.add_argument(
        "--source-depth-m", type=float, default=278.0, help="Analytic source depth"
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

        # Compute best-fit scale
        sem_flat = sem_disp.ravel()
        ref_flat = ref_disp.ravel()
        scale = float(np.dot(ref_flat, sem_flat) / (np.dot(sem_flat, sem_flat) + 1e-30))
        scaled_sem = sem_disp * scale

        # Metrics
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
                "sem_disp": sem_disp,
                "ref_disp": ref_disp,
                "scaled_sem_disp": scaled_sem,
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
        print(f"\n  {label} ({len(group)} points):")
        print(
            f"    rel_l2:  mean={np.mean(rel_l2s):.4f}, median={np.median(rel_l2s):.4f}, min={np.min(rel_l2s):.4f}, max={np.max(rel_l2s):.4f}"
        )
        print(
            f"    corr:    mean={np.mean(corrs):.4f}, median={np.median(corrs):.4f}, min={np.min(corrs):.4f}, max={np.max(corrs):.4f}"
        )

    # Save results
    save_dict = {
        "positions": np.array([r["position"] for r in results]),
        "is_vertex": np.array([r["is_vertex"] for r in results]),
        "interpolated": np.array([r["interpolated"] for r in results]),
        "scales": np.array([r["scale"] for r in results]),
        "rel_l2": np.array([r["rel_l2"] for r in results]),
        "correlation": np.array([r["correlation"] for r in results]),
    }
    np.savez(args.output, **save_dict)
    print(f"\nSaved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
