#!/usr/bin/env python3
"""Compare viscoelastic greenfun tiles with elastic reference greenfun tiles.

For elastic-limit regression: the viscoelastic solver with Q→∞ should
produce displacement tensors nearly identical to the pure elastic solver.

Usage:
    python3 elastic_limit_compare.py <elastic_greenfun_dir> <viscoelastic_greenfun_dir>
                                     [--tol 0.01] [--output comparison.npz]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def _get_dataset(h5file: h5py.File, path: str) -> h5py.Dataset:
    """Get a Dataset from an HDF5 file, exiting if not found or not a Dataset."""
    item = h5file.get(path)
    if not isinstance(item, h5py.Dataset):
        print(f"FAIL: {path} not found or not a Dataset", file=sys.stderr)
        sys.exit(1)
    return item


def compute_rel_l2(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute relative L2 error: ||test - ref||_2 / ||ref||_2."""
    diff = test - ref
    diff_norm = float(np.sqrt(np.sum(diff**2)))
    ref_norm = float(np.sqrt(np.sum(ref**2)))
    if ref_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")
    return diff_norm / ref_norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elastic-limit regression: compare viscoelastic vs elastic greenfun"
    )
    parser.add_argument("elastic_dir", help="Path to elastic greenfun/ directory")
    parser.add_argument("viscoelastic_dir", help="Path to viscoelastic greenfun/ directory")
    parser.add_argument(
        "--tol",
        type=float,
        default=0.01,
        help="Max acceptable relative L2 error (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Optional .npz output path for comparison data"
    )
    args = parser.parse_args()

    elastic_dir = Path(args.elastic_dir)
    viscoelastic_dir = Path(args.viscoelastic_dir)

    if not elastic_dir.is_dir():
        print(f"FAIL: elastic greenfun dir not found: {elastic_dir}", file=sys.stderr)
        sys.exit(1)
    if not viscoelastic_dir.is_dir():
        print(f"FAIL: viscoelastic greenfun dir not found: {viscoelastic_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover tiles
    elastic_tiles = sorted(glob.glob(str(elastic_dir / "tile_*.h5")))
    visco_tiles = sorted(glob.glob(str(viscoelastic_dir / "tile_*.h5")))

    if not elastic_tiles:
        print(f"FAIL: no tile_*.h5 files in {elastic_dir}", file=sys.stderr)
        sys.exit(1)
    if not visco_tiles:
        print(f"FAIL: no tile_*.h5 files in {viscoelastic_dir}", file=sys.stderr)
        sys.exit(1)

    # Match tiles by basename
    elastic_by_name = {os.path.basename(t): t for t in elastic_tiles}
    visco_by_name = {os.path.basename(t): t for t in visco_tiles}

    common = sorted(set(elastic_by_name.keys()) & set(visco_by_name.keys()))
    if not common:
        print(
            "FAIL: no common tile filenames between elastic and viscoelastic dirs", file=sys.stderr
        )
        sys.exit(1)

    print(f"  elastic tiles: {len(elastic_tiles)}, viscoelastic tiles: {len(visco_tiles)}")
    print(f"  common tiles: {len(common)}")

    max_rel_l2 = 0.0
    per_component = np.zeros(9)  # 3x3 displacement tensor flattened
    component_max = np.zeros(9)
    tile_errors: dict[str, float] = {}

    for basename in common:
        with (
            h5py.File(elastic_by_name[basename], "r") as f_el,
            h5py.File(visco_by_name[basename], "r") as f_visco,
        ):
            disp_el = np.asarray(_get_dataset(f_el, "/field/displacement_tensor"))
            disp_visco = np.asarray(_get_dataset(f_visco, "/field/displacement_tensor"))

        if disp_el.shape != disp_visco.shape:
            print(
                f"  WARN: shape mismatch {basename}: "
                f"elastic={disp_el.shape} viscoelastic={disp_visco.shape}"
            )
            continue

        tile_l2 = compute_rel_l2(disp_el, disp_visco)
        tile_errors[basename] = tile_l2
        max_rel_l2 = max(max_rel_l2, tile_l2)

        # Per-component relative L2
        if disp_el.ndim >= 4:
            # Shape: [n_vertices, 3, 3] or similar
            for c in range(min(9, np.prod(disp_el.shape[1:]))):
                idx = np.unravel_index(c, disp_el.shape[1:])
                ref_c = disp_el[(slice(None),) + idx]
                test_c = disp_visco[(slice(None),) + idx]
                comp_l2 = compute_rel_l2(ref_c, test_c)
                component_max[c] = max(component_max[c], comp_l2)

    # Summary
    print(f"\n  Tiles compared: {len(tile_errors)}")
    print(f"  Max relative L2 error: {max_rel_l2:.6e}")
    print(f"  Tolerance: {args.tol:.6e}")

    # Per-component summary
    if component_max.sum() > 0:
        print(f"  Max per-component relative L2 errors: {component_max}")

    # Save comparison data
    if args.output:
        np.savez(
            args.output,
            max_rel_l2=max_rel_l2,
            tile_errors=tile_errors,
            tol=args.tol,
            elastic_dir=str(elastic_dir),
            viscoelastic_dir=str(viscoelastic_dir),
        )
        print(f"  Comparison saved to {args.output}")

    # Verdict
    if max_rel_l2 <= args.tol:
        print(f"\n  ELASTIC LIMIT CHECK — PASSED (rel_l2={max_rel_l2:.4e} <= tol={args.tol:.0e})")
    else:
        print(
            f"\n  ELASTIC LIMIT CHECK — FAILED (rel_l2={max_rel_l2:.4e} > tol={args.tol:.0e})",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
