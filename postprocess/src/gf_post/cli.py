"""CLI entry point for the postprocessing pipeline.

Usage:
    gf-postprocess mesh.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
"""

import glob
import os
import sys
import time

import click
import numpy as np

from gf_post.assembly import assemble_greens_tensor
from gf_post.reader import GeometryReader, merge_records
from gf_post.writer import GFWriter


def _discover_record_files(record_dir: str) -> list[str]:
    """Find all record_{r}.h5 files in a directory, sorted by rank."""
    pattern = os.path.join(record_dir, "record_*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No record files found in {record_dir}")
    return files


@click.command()
@click.argument("mesh", type=click.Path(exists=True))
@click.option(
    "--fx",
    required=True,
    type=click.Path(exists=True),
    help="Directory with fx-direction record files",
)
@click.option(
    "--fy",
    required=True,
    type=click.Path(exists=True),
    help="Directory with fy-direction record files",
)
@click.option(
    "--fz",
    required=True,
    type=click.Path(exists=True),
    help="Directory with fz-direction record files",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    default="greenfun",
    help="Output directory for Green's function tile files",
)
@click.option("--tile-elems", type=int, default=100, help="Maximum elements per spatial tile")
def main(mesh, fx, fy, fz, output_dir, tile_elems):
    """Extract strain Green's functions from SEM checkpoint files.

    Reads strain checkpoints from three force-direction forward runs,
    merges per-rank records, assembles the full 3×6 Green's tensor at
    every GLL node, and writes spatially tiled HDF5 output.

    No receiver positions needed — output is the full GLL-node field.
    """
    start = time.time()
    print("[postprocess] Starting...", file=sys.stderr)

    # Load mesh geometry
    print(f"[postprocess] Reading mesh geometry from {mesh}", file=sys.stderr)
    with GeometryReader(mesh) as geo:
        gll_coords = geo.coords
        n_cell = geo.n_cell
        ngll = geo.ngll

    # Discover record files
    fx_files = _discover_record_files(fx)
    fy_files = _discover_record_files(fy)
    fz_files = _discover_record_files(fz)
    print(
        f"[postprocess] Found {len(fx_files)} fx, {len(fy_files)} fy, "
        f"{len(fz_files)} fz record files",
        file=sys.stderr,
    )

    # Merge records from each direction
    print("[postprocess] Merging fx records...", file=sys.stderr)
    strain_fx, info_fx = merge_records(fx_files, n_cell)

    print("[postprocess] Merging fy records...", file=sys.stderr)
    strain_fy, info_fy = merge_records(fy_files, n_cell)

    print("[postprocess] Merging fz records...", file=sys.stderr)
    strain_fz, info_fz = merge_records(fz_files, n_cell)

    # Time alignment validation
    for name, info in [("fx", info_fx), ("fy", info_fy), ("fz", info_fz)]:
        if info["dt"] != info_fx["dt"] or info["nsteps"] != info_fx["nsteps"]:
            raise ValueError(
                f"Time alignment mismatch: {name} has dt={info['dt']} "
                f"nsteps={info['nsteps']}, expected dt={info_fx['dt']} "
                f"nsteps={info_fx['nsteps']}"
            )

    dt = info_fx["dt"]
    nt = strain_fx.shape[0]
    time_arr = np.arange(nt) * dt * info_fx["checkpoint_interval"]

    # Assemble Green's tensor at all GLL nodes
    # strain_fx shape: [nt, n_cell, NGLL, NGLL, NGLL, 6]
    # greens_tensor shape: [nt, n_cell, NGLL, NGLL, NGLL, 6, 3]
    print("[postprocess] Assembling Green's tensor at all GLL nodes...", file=sys.stderr)
    greens = assemble_greens_tensor({"fx": strain_fx, "fy": strain_fy, "fz": strain_fz})

    # Write output — tiled by element range
    print(f"[postprocess] Writing Green's function tiles to {output_dir}...", file=sys.stderr)
    tiles = GFWriter.write(output_dir, gll_coords, time_arr, dt, greens, tile_elems)

    elapsed = time.time() - start
    n_tiles = len(tiles)
    print(
        f"[postprocess] Done in {elapsed:.1f}s — {n_tiles} tile(s), {n_cell} element(s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
