#!/usr/bin/env python3
"""Compare strain fields between CUDA and MPI-CPU elastic solver runs.

Aligns recorded strain on common GLL node coordinates at selected time
steps and reports relative L2 error and Pearson correlation.

NOTE: the CUDA (nompi) and MPI runs use DIFFERENT global node numberings
(field/cell vs partition numbering in model.h5), so alignment must use
physical coordinates, not gll_node_ids.

CUDA run:   wavefields/x/record_0_{step}.h5                (single rank)
MPI run:    tmp/mpi_run/wavefields/x/record_{r}_{step}.h5  (16 ranks)

Usage:
    python3 compare_solvers.py [step ...]        # default: 400 700
"""

from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_DIR = os.path.join(CASE_DIR, "wavefields", "x")
MPI_DIR = os.path.join(CASE_DIR, "tmp", "mpi_run", "wavefields", "x")

N_STRAIN_COMPONENTS = 6


def _read_record(record_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read (strain, cell_gll_node_index, gll_node_coords) from one record file."""
    with h5py.File(record_path, "r") as record_file:
        strain_obj = record_file["strain"]
        index_obj = record_file["cell_gll_node_index"]
        coords_obj = record_file["gll_node_coords"]
        assert isinstance(strain_obj, h5py.Dataset)
        assert isinstance(index_obj, h5py.Dataset)
        assert isinstance(coords_obj, h5py.Dataset)
        strain = np.asarray(strain_obj[0], dtype=np.float64)  # (n_rec_cell, 125, 6)
        cell_gll_index = np.asarray(index_obj[:], dtype=np.int64)  # (n_rec_cell, 125)
        node_coords = np.asarray(coords_obj[:], dtype=np.float64)  # (n_unique, 3)
    return strain, cell_gll_index, node_coords


def load_strain_by_node_coords(pattern: str) -> dict[tuple, np.ndarray]:
    """Load record files and map rounded GLL node coordinates -> strain vector.

    Nodes shared across ranks (element interfaces) appear in multiple
    files; duplicated values are averaged (they should be identical).
    Coordinates are rounded to 1 mm to make them hash-stable.
    """
    strain_sum: dict[tuple, np.ndarray] = {}
    strain_count: dict[tuple, int] = {}
    for record_path in sorted(glob.glob(pattern)):
        strain, cell_gll_index, node_coords = _read_record(record_path)
        flat_strain = strain.reshape(-1, N_STRAIN_COMPONENTS)
        flat_coords = np.round(node_coords[cell_gll_index.reshape(-1)], 3)
        for xyz, node_strain in zip(flat_coords, flat_strain):
            key = (xyz[0], xyz[1], xyz[2])
            if key in strain_sum:
                strain_sum[key] += node_strain
                strain_count[key] += 1
            else:
                strain_sum[key] = node_strain
                strain_count[key] = 1
    return {key: strain_sum[key] / strain_count[key] for key in strain_sum}


def compare_step(step: int) -> tuple[float, float, int]:
    """Return (rel_l2, pearson_corr, n_common_nodes) for one time step."""
    cuda_map = load_strain_by_node_coords(os.path.join(CUDA_DIR, f"record_0_{step}.h5"))
    mpi_map = load_strain_by_node_coords(os.path.join(MPI_DIR, f"record_*_{step}.h5"))

    common_ids = sorted(cuda_map.keys() & mpi_map.keys())
    if not common_ids:
        raise RuntimeError(f"step {step}: no common recorded GLL nodes")

    cuda_strain = np.array([cuda_map[i] for i in common_ids])
    mpi_strain = np.array([mpi_map[i] for i in common_ids])

    cuda_norm = np.linalg.norm(cuda_strain)
    if cuda_norm == 0.0:
        raise RuntimeError(f"step {step}: zero CUDA strain norm")

    diff = mpi_strain - cuda_strain
    try:
        rel_l2 = float(np.linalg.norm(diff) / cuda_norm)
        pearson_corr = float(np.corrcoef(cuda_strain.reshape(-1), mpi_strain.reshape(-1))[0, 1])
    except (ValueError, FloatingPointError) as exc:
        raise RuntimeError(f"step {step}: metric computation failed: {exc}") from exc
    return rel_l2, pearson_corr, len(common_ids)


def main() -> int:
    try:
        steps = [int(arg) for arg in sys.argv[1:]] or [400, 700]
    except ValueError as exc:
        print(f"invalid step argument: {exc}", file=sys.stderr)
        return 2

    print(f"{'step':>6} {'n_common':>10} {'rel_l2':>12} {'corr':>12}")
    worst_rel_l2 = 0.0
    worst_corr = 1.0
    for step in steps:
        rel_l2, pearson_corr, n_common = compare_step(step)
        worst_rel_l2 = max(worst_rel_l2, rel_l2)
        worst_corr = min(worst_corr, pearson_corr)
        print(f"{step:>6} {n_common:>10} {rel_l2:>12.6e} {pearson_corr:>12.8f}")

    passed = worst_rel_l2 < 0.01 and worst_corr > 0.999
    print()
    print(
        f"VERDICT: {'CONSISTENT' if passed else 'INCONSISTENT'} "
        f"(worst rel_l2={worst_rel_l2:.6e}, worst corr={worst_corr:.8f}; "
        f"thresholds: rel_l2<0.01, corr>0.999)"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
