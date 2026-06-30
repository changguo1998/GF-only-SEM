"""CLI entry point for the postprocessing pipeline.

Usage:
    gf-postprocess model.h5 config.h5 --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/
"""

import glob
import os
import sys
import time

import click
import numpy as np

from gf_post.assembly import assemble_greens_tensor
from gf_post.reader import ConfigReader, GeometryReader, merge_vertex_records
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
@click.argument("config", type=click.Path(exists=True))
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
def main(mesh, config, fx, fy, fz, output_dir):
    """Extract strain Green's functions from SEM record files.

    Reads mesh-vertex strain records from three force-direction forward runs,
    merges per-rank records, assembles the full 3x6 Green's tensor at
    every recorded mesh vertex, and writes spatially-tiled HDF5 output.

    Tile sizes come from config.h5 /simulation/tilex_elements and tiley_elements (element counts).
    """
    start = time.time()
    print("[postprocess] Starting...", file=sys.stderr)

    # Read config
    print(f"[postprocess] Reading config from {config}", file=sys.stderr)
    with ConfigReader(config) as cfg:
        nx_elements = cfg.nx_elements
        ny_elements = cfg.ny_elements
        pml_thickness = cfg.pml_thickness
        tilex_elements = cfg.tilex_elements
        tiley_elements = cfg.tiley_elements
        record_depth_max_m = cfg.record_depth_max_m
        record_depth_actual_m = cfg.record_depth_actual_m
        solver_dt = cfg.solver_dt
        output_dt_s = cfg.output_dt_s

    # Load mesh geometry
    print(f"[postprocess] Reading mesh geometry from {mesh}", file=sys.stderr)
    with GeometryReader(mesh) as geo:
        vertex_coords = geo.vertex_coords
        n_vertex = geo.n_vertex
        domain_bounds = geo.domain_bounds

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
    strain_fx, mask_fx = merge_vertex_records(fx_files, n_vertex)

    print("[postprocess] Merging fy records...", file=sys.stderr)
    strain_fy, mask_fy = merge_vertex_records(fy_files, n_vertex)

    print("[postprocess] Merging fz records...", file=sys.stderr)
    strain_fz, mask_fz = merge_vertex_records(fz_files, n_vertex)

    # Consistency check: same recorded vertices across directions
    if not np.array_equal(mask_fx, mask_fy) or not np.array_equal(mask_fx, mask_fz):
        print(
            "[postprocess] WARNING: recorded vertex sets differ across directions", file=sys.stderr
        )

    # Use combined mask of all recorded vertices
    vertex_mask = mask_fx & mask_fy & mask_fz
    recorded_indices = np.where(vertex_mask)[0]
    recorded_ids = np.arange(1, n_vertex + 1, dtype=np.int64)[recorded_indices]
    print(f"[postprocess] {len(recorded_indices)}/{n_vertex} vertices recorded", file=sys.stderr)

    n_snapshots = strain_fx.shape[0]
    time_arr = np.arange(n_snapshots) * output_dt_s

    # Subset to recorded vertices only
    strain_fx = strain_fx[:, recorded_indices, :]
    strain_fy = strain_fy[:, recorded_indices, :]
    strain_fz = strain_fz[:, recorded_indices, :]

    # Assemble Green's tensor at recorded vertices
    print("[postprocess] Assembling Green's tensor...", file=sys.stderr)
    greens = assemble_greens_tensor({"fx": strain_fx, "fy": strain_fy, "fz": strain_fz})

    # Write output — spatially tiled
    print(f"[postprocess] Writing Green's function tiles to {output_dir}...", file=sys.stderr)
    tiles = GFWriter.write(
        output_dir,
        vertex_coords[recorded_indices],
        recorded_ids,
        time_arr,
        solver_dt,
        greens,
        nx_elements,
        ny_elements,
        pml_thickness,
        tilex_elements,
        tiley_elements,
        domain_bounds,
        record_depth_max_m=record_depth_max_m,
        record_depth_actual_m=record_depth_actual_m,
    )

    elapsed = time.time() - start
    n_tiles = len(tiles)
    print(
        f"[postprocess] Done in {elapsed:.1f}s — {n_tiles} tile(s), "
        f"{len(recorded_indices)} recorded vertex(ices)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
