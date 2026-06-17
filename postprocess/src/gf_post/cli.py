"""CLI entry point for the postprocessing pipeline.

Usage:
    gf-postprocess mesh.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ receivers.csv -o greenfun/
"""

import csv
import glob
import os
import sys
import time
from pathlib import Path

import click
import numpy as np

from gf_post.reader import RecordReader, GeometryReader, merge_records
from gf_post.index import ElementIndex
from gf_post.search import find_containing_element
from gf_post.interpolate import interpolate_strain
from gf_post.assembly import assemble_greens_tensor
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
@click.option("--fx", required=True, type=click.Path(exists=True),
              help="Directory with fx-direction record files")
@click.option("--fy", required=True, type=click.Path(exists=True),
              help="Directory with fy-direction record files")
@click.option("--fz", required=True, type=click.Path(exists=True),
              help="Directory with fz-direction record files")
@click.argument("receivers_csv", type=click.Path(exists=True))
@click.option("-o", "--output-dir", type=click.Path(), default="greenfun",
              help="Output directory for Green's function tiles")
@click.option("--tile-size", type=int, default=1000,
              help="Maximum receivers per tile")
def main(mesh, fx, fy, fz, receivers_csv, output_dir, tile_size):
    """Extract strain Green's functions from SEM checkpoint files.

    Reads strain checkpoints from three force-direction forward runs,
    locates receivers in the mesh, interpolates strain, and writes
    Green's function tile files.
    """
    start = time.time()
    print(f"[postprocess] Starting...", file=sys.stderr)

    # Load mesh geometry
    print(f"[postprocess] Reading mesh geometry from {mesh}", file=sys.stderr)
    with GeometryReader(mesh) as geo:
        gll_coords = geo.coords
        dxi_dx = geo.dxi_dx
        is_pml = geo.is_pml
        n_cell = geo.n_cell
        ngll = geo.ngll
    
    # Discover record files
    fx_files = _discover_record_files(fx)
    fy_files = _discover_record_files(fy)
    fz_files = _discover_record_files(fz)
    print(f"[postprocess] Found {len(fx_files)} fx, {len(fy_files)} fy, {len(fz_files)} fz record files", file=sys.stderr)
    
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
                f"Time alignment mismatch: {name} has dt={info['dt']} nsteps={info['nsteps']}, "
                f"expected dt={info_fx['dt']} nsteps={info_fx['nsteps']}"
            )
    
    dt = info_fx["dt"]
    n_records = strain_fx.shape[0]
    nt = n_records
    time_arr = np.arange(nt) * dt * info_fx["checkpoint_interval"]
    
    # Load receivers
    print("[postprocess] Loading receivers...", file=sys.stderr)
    receivers = []
    names = []
    with open(receivers_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            receivers.append([float(row["x"]), float(row["y"]), float(row["z"])])
            names.append(row["name"])
    receivers = np.array(receivers)
    n_recv = len(receivers)
    
    # Build spatial index
    print("[postprocess] Building spatial index...", file=sys.stderr)
    idx = ElementIndex(gll_coords, is_pml)
    
    # For each receiver, find containing element and interpolate
    print("[postprocess] Interpolating strain at receivers...", file=sys.stderr)
    waveforms_fx = np.zeros((nt, n_recv, 6), dtype=np.float64)
    waveforms_fy = np.zeros((nt, n_recv, 6), dtype=np.float64)
    waveforms_fz = np.zeros((nt, n_recv, 6), dtype=np.float64)
    
    for i in range(n_recv):
        point = receivers[i]
        
        # Query KD-tree for candidate elements
        candidates, _ = idx.query(point, k=20)
        
        # Find containing element
        elem_id, xi, eta, zeta = find_containing_element(
            point, candidates, gll_coords, dxi_dx
        )
        
        # Interpolate strain for each timestep
        for t in range(nt):
            waveforms_fx[t, i] = interpolate_strain(
                strain_fx[t, elem_id - 1], xi, eta, zeta
            )
            waveforms_fy[t, i] = interpolate_strain(
                strain_fy[t, elem_id - 1], xi, eta, zeta
            )
            waveforms_fz[t, i] = interpolate_strain(
                strain_fz[t, elem_id - 1], xi, eta, zeta
            )
    
    # Write output
    print(f"[postprocess] Writing Green's function tiles to {output_dir}...", file=sys.stderr)
    tiles = GFWriter.write(output_dir, receivers, time_arr,
                           {"fx": waveforms_fx, "fy": waveforms_fy, "fz": waveforms_fz},
                           tile_size)
    
    elapsed = time.time() - start
    print(f"[postprocess] Done in {elapsed:.1f}s — {len(tiles)} tile(s), {n_recv} receiver(s)", file=sys.stderr)


if __name__ == "__main__":
    main()