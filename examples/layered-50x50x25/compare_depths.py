"""Compare source-depth SEM results with the surface case and PyFK."""

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

CASE = Path(__file__).resolve().parent
ROOT = CASE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CASE))

import config  # noqa: E402

from greenfun.library import GreenFunctionLibrary  # noqa: E402

# Reuse the PyFK implementation while supplying this case's layered model.
LAYERED_REFERENCE = ROOT / "examples/layer-shallow-source"
sys.path.insert(0, str(LAYERED_REFERENCE))
import reference as pyfk_reference  # noqa: E402

pyfk_reference.LAYER_MODEL = config.LAYER_MODEL
pyfk_reference.DK = config.DK
pyfk_reference.SMTH = config.SMTH
pyfk_reference.PMIN = config.PMIN
pyfk_reference.PMAX = config.PMAX
pyfk_reference.KMAX = config.KMAX
pyfk_reference.SAMPLES_BEFORE_FIRST_ARRIVAL = config.SAMPLES_BEFORE_FIRST_ARRIVAL

DEPTHS_M = (0, 1, 10, 100, 500, 900, 1100)
REFERENCE_OFFSETS_M = (2000.0, 14000.0)


def metrics(reference, candidate, time_s, end_time_s):
    mask = time_s < end_time_s
    reference = np.asarray(reference[mask], dtype=np.float64).ravel()
    candidate = np.asarray(candidate[mask], dtype=np.float64).ravel()
    reference_norm = np.linalg.norm(reference)
    candidate_norm = np.linalg.norm(candidate)
    scale = float(np.dot(candidate, reference) / np.dot(reference, reference))
    fitted = float(
        np.linalg.norm(candidate - scale * reference)
        / max(candidate_norm, abs(scale) * reference_norm, 1.0e-300)
    )
    correlation = float(np.corrcoef(reference, candidate)[0, 1])
    raw = float(np.linalg.norm(candidate - reference) / reference_norm)
    return correlation, scale, raw, fitted


def load_archive(depth_m):
    archive = CASE / "depth_sweep" / f"depth_{depth_m:04d}m"
    if not (archive / "complete.txt").is_file():
        raise FileNotFoundError(f"Incomplete source-depth run: {archive}")
    library = GreenFunctionLibrary(archive / "greenfun")
    with h5py.File(archive / "config.h5", "r") as stream:
        source_attributes = stream["source"].attrs
        source_xyz = np.array(
            [source_attributes["x"], source_attributes["y"], source_attributes["z"]],
            dtype=np.float64,
        )
        source_time_s = np.asarray(stream["source/stf_t"], dtype=np.float64)
        source_values = np.asarray(stream["source/stf_values"], dtype=np.float64)
        simulation = stream["simulation"].attrs
        output_dt_s = float(simulation["output_dt_s"])
    tile_path = next((archive / "greenfun").glob("tile_*.h5"))
    with h5py.File(tile_path, "r") as stream:
        time_s = np.asarray(stream["time/t"], dtype=np.float64)
    if not np.isclose(source_xyz[2], depth_m):
        raise ValueError(f"Archived source depth {source_xyz[2]} != {depth_m}")
    return archive, library, source_xyz, time_s, source_time_s, source_values, output_dt_s


def main():
    archives = {depth: load_archive(depth) for depth in DEPTHS_M}
    surface = archives[0]
    rows = []
    for depth_m, archive_data in archives.items():
        _, library, source_xyz, time_s, source_time_s, source_values, output_dt_s = archive_data
        surface_library = surface[1]
        surface_source = surface[2]
        if not np.array_equal(time_s, surface[3]):
            raise ValueError("Source-depth cases use different output time grids")

        for offset_m, end_time_s in zip(
            config.validation_offsets_m, config.validation_end_times_s
        ):
            receiver_xyz = source_xyz + [offset_m, 0.0, -source_xyz[2]]
            surface_receiver_xyz = surface_source + [offset_m, 0.0, 0.0]
            result = library.query(receiver_xyz, source_xyz, quantity="displacement")
            baseline = surface_library.query(
                surface_receiver_xyz, surface_source, quantity="displacement"
            )
            if result.displacement is None or baseline.displacement is None:
                raise ValueError("Missing displacement in archived Green function")
            if (
                not np.isfinite(result.displacement).all()
                or not np.isfinite(baseline.displacement).all()
            ):
                raise ValueError("Nonfinite SEM displacement")

            corr, scale, raw, fitted = metrics(
                baseline.displacement, result.displacement, time_s, end_time_s
            )
            rows.append(
                dict(
                    comparison="against_surface_sem",
                    source_depth_m=depth_m,
                    offset_m=offset_m,
                    end_time_s=end_time_s,
                    correlation=corr,
                    scale=scale,
                    raw_relative_l2=raw,
                    fitted_relative_l2=fitted,
                )
            )

            # Check theoretical agreement at both ends of the distance range.
            if offset_m not in REFERENCE_OFFSETS_M:
                continue
            pyfk_source_xyz = source_xyz.copy()
            if pyfk_source_xyz[2] == 0.0:
                # PyFK rejects a point source exactly on the free-surface interface.
                pyfk_source_xyz[2] = 1.0
            reference_step = pyfk_reference.compute_green_tensor(
                source_xyz_m=pyfk_source_xyz,
                receiver_xyz_m=receiver_xyz,
                time=time_s,
                force_amplitude=config.FORCE_AMPLITUDE,
            )
            source_on_output_grid = np.interp(time_s, source_time_s, source_values)
            reference_displacement = np.zeros_like(reference_step)
            for output_axis in range(3):
                for force_axis in range(3):
                    convolution = (
                        np.convolve(
                            reference_step[:, output_axis, force_axis],
                            source_on_output_grid,
                            mode="full",
                        )
                        * output_dt_s
                    )
                    reference_displacement[:, output_axis, force_axis] = convolution[: len(time_s)]
            ref_corr, ref_scale, ref_raw, ref_fitted = metrics(
                reference_displacement, result.displacement, time_s, end_time_s
            )
            rows.append(
                dict(
                    comparison="against_pyfk",
                    source_depth_m=depth_m,
                    offset_m=offset_m,
                    end_time_s=end_time_s,
                    correlation=ref_corr,
                    scale=ref_scale,
                    raw_relative_l2=ref_raw,
                    fitted_relative_l2=ref_fitted,
                )
            )
            print(
                f"depth={depth_m}m offset={offset_m / 1000:g}km "
                f"PyFK corr={ref_corr:.6f} scale={ref_scale:.6f} "
                f"raw_l2={ref_raw:.6f} fitted_l2={ref_fitted:.6f}",
                flush=True,
            )

        print(f"depth={depth_m}m comparison with surface SEM complete", flush=True)

    output_path = ROOT / "docs/test-reports/data/layered-50x50x25-source-depth.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
