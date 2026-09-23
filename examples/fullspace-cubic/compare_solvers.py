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
    basename = os.path.basename(record_path)
    rank = int(basename.split("_")[1])
    with h5py.File(record_path, "r") as record_file:
        strain_obj = record_file["strain"]
        assert isinstance(strain_obj, h5py.Dataset)
        strain = np.asarray(strain_obj[0], dtype=np.float64)  # (n_rec_cell, 125, 6)
        partition_start = int(record_file.attrs.get("source_partition_start", rank))
        partition_count = int(record_file.attrs.get("source_partition_count", 1))

    case_dir = os.path.dirname(os.path.dirname(os.path.dirname(record_path)))
    global_to_merged: dict[int, int] = {}
    merged_coordinates: list[np.ndarray] = []
    merged_cell_indexes: list[np.ndarray] = []
    for partition_index in range(partition_start, partition_start + partition_count):
        partition_path = os.path.join(case_dir, "partitions", f"partition_{partition_index}.h5")
        with h5py.File(partition_path, "r") as partition:
            if "recording" not in partition:
                continue
            recording = partition["recording"]
            node_ids = np.asarray(recording["gll_node_ids"], dtype=np.int64)
            coordinates = np.asarray(recording["gll_node_coords"], dtype=np.float64)
            cell_indexes = np.asarray(recording["cell_gll_node_index"], dtype=np.int64)

        partition_to_merged = np.empty(node_ids.size, dtype=np.int64)
        for node_index, node_id in enumerate(node_ids):
            merged_index = global_to_merged.get(int(node_id))
            if merged_index is None:
                merged_index = len(merged_coordinates)
                global_to_merged[int(node_id)] = merged_index
                merged_coordinates.append(coordinates[node_index])
            partition_to_merged[node_index] = merged_index
        merged_cell_indexes.append(partition_to_merged[cell_indexes.reshape(-1)])

    node_coords = np.asarray(merged_coordinates, dtype=np.float64)
    cell_gll_index = np.concatenate(merged_cell_indexes).reshape(strain.shape[:2])
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
