#!/usr/bin/env python3
"""Inject SLS attenuation fields into model.h5.

For elastic-limit regression: use very large Q values (q_mu=1e9, q_kappa=1e9)
so the SLS solver code path is exercised but attenuation is negligible.

Usage:
    python3 attenuation_inject.py model.h5 [--q-mu 1e9] [--q-kappa 1e9] [--n-sls 3] [--f0 2.0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Add project root to path
_project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_project_root))

from preprocess.attenuation import write_attenuation_to_model  # noqa: E402


def _get_dataset(h5file: h5py.File, path: str) -> h5py.Dataset:
    """Get a Dataset from an HDF5 file, exiting if not found or not a Dataset."""
    item = h5file.get(path)
    if not isinstance(item, h5py.Dataset):
        print(f"FAIL: {path} not found or not a Dataset", file=sys.stderr)
        sys.exit(1)
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject SLS attenuation into model.h5")
    parser.add_argument("model_path", help="Path to model.h5")
    parser.add_argument(
        "--q-mu", type=float, default=1.0e9, help="Shear quality factor (default: 1e9)"
    )
    parser.add_argument(
        "--q-kappa", type=float, default=1.0e9, help="Bulk quality factor (default: 1e9)"
    )
    parser.add_argument(
        "--n-sls", type=int, default=3, help="Number of SLS mechanisms (default: 3)"
    )
    parser.add_argument(
        "--f0", type=float, default=2.0, help="Reference frequency in Hz (default: 2.0)"
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"FAIL: model.h5 not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    # Detect n_cell and NGLL from coords dataset
    # C++ preprocessor writes to field/element/, Python writes to field/cell/
    with h5py.File(model_path, "r") as f:
        coords_path = "field/element/coords"
        if coords_path not in f:
            coords_path = "field/cell/coords"
        coords_ds = _get_dataset(f, coords_path)
        shape = coords_ds.shape  # [n_cell, NGLL, NGLL, NGLL, 3]
        n_cell = shape[0]
        ngll = shape[1]
        if coords_ds.dtype == np.float32:
            print("  model dtype=float32 (C++ preprocessor)")

    print(f"  n_cell={n_cell}, NGLL={ngll}, n_sls={args.n_sls}, f0={args.f0} Hz")
    print(f"  q_mu={args.q_mu:.1e}, q_kappa={args.q_kappa:.1e}")

    # Create constant Q arrays
    q_mu = np.full((n_cell, ngll, ngll, ngll), args.q_mu, dtype=np.float64)
    q_kappa = np.full((n_cell, ngll, ngll, ngll), args.q_kappa, dtype=np.float64)

    # Write attenuation to model.h5
    write_attenuation_to_model(str(model_path), q_kappa, q_mu, ngll, args.n_sls, args.f0)

    # Verify
    with h5py.File(model_path, "r") as f:
        field_group = "element" if "field/element/tau_sigma" in f else "cell"
        for name in ["tau_sigma", "tau_epsilon_mu", "tau_epsilon_kappa", "q_mu", "q_kappa"]:
            path = f"/field/{field_group}/{name}"
            ds = _get_dataset(f, path)
            print(f"  {path}: shape={ds.shape}, dtype={ds.dtype}")

    print("  attenuation injection — DONE")


if __name__ == "__main__":
    main()
