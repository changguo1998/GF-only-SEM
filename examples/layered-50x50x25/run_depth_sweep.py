"""Run the large aligned-interface model at selected source depths."""

import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASE = Path(__file__).resolve().parent
RESULTS = CASE / "depth_sweep"
DEPTHS_M = (0, 1, 10, 100, 500, 900, 1100)
PYTHON = ROOT / ".venv/bin/python"
PREPROCESS_BUILD = ROOT / "build/preprocess-layered-50x50x25"
PREPROCESS_BIN = PREPROCESS_BUILD / "bin/gf_preprocess"


def run(command, log_path, environment, cwd=CASE):
    started = time.monotonic()
    with log_path.open("w") as log:
        subprocess.run(
            command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    elapsed = time.monotonic() - started
    print(f"{log_path.name}: {elapsed:.1f}s", flush=True)
    return elapsed


def main():
    available_bytes = shutil.disk_usage(ROOT).free
    required_bytes = 360 * 1024**3
    if available_bytes < required_bytes:
        raise RuntimeError(
            f"Need {required_bytes / 1024**3:.0f} GiB free for one run; "
            f"found {available_bytes / 1024**3:.0f} GiB"
        )

    RESULTS.mkdir(exist_ok=True)
    for depth_m in DEPTHS_M:
        destination = RESULTS / f"depth_{depth_m:04d}m"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing case: {destination}")

    environment = dict(os.environ, OMP_NUM_THREADS="8", PYTHONPATH=str(ROOT))
    run(
        [
            "cmake",
            "-S",
            str(ROOT / "preprocess/cpp"),
            "-B",
            str(PREPROCESS_BUILD),
            f"-DGF_USER_CONFIG={CASE / 'config_user.cpp'}",
            f"-DGF_RUNTIME_OUTPUT_DIRECTORY={PREPROCESS_BIN.parent}",
        ],
        CASE / "build_configure.log",
        environment,
    )
    run(
        ["cmake", "--build", str(PREPROCESS_BUILD), "--target", "gf_preprocess", "-j8"],
        CASE / "build.log",
        environment,
    )

    for depth_m in DEPTHS_M:
        destination = RESULTS / f"depth_{depth_m:04d}m"
        destination.mkdir()
        run_environment = dict(
            environment, GF_SOURCE_DEPTH_M="-1" if depth_m == 0 else str(depth_m)
        )
        print(f"START source depth={depth_m} m", flush=True)
        case_started = time.monotonic()

        # Rebuild the topology-only mesh so each preprocessor run starts clean.
        run(
            [str(PYTHON), str(ROOT / "examples/halfspace-50x50x25/mesh_gen.py")],
            destination / "mesh.log",
            run_environment,
        )
        run(
            [
                str(PREPROCESS_BIN),
                "run",
                "model.h5",
                "--N",
                "4",
                "--cfl-safety",
                "0.5",
                "--nx",
                "50",
                "--ny",
                "50",
                "--pml-xmin",
                "5",
                "--pml-xmax",
                "5",
                "--pml-ymin",
                "5",
                "--pml-ymax",
                "5",
                "--pml-zmax",
                "5",
                "--n-ranks",
                "16",
            ],
            destination / "preprocess.log",
            run_environment,
        )

        # Verify that preprocessing stored the requested source coordinate.
        verify_source = (
            "import h5py, numpy as np; "
            "f=h5py.File('config.h5','r'); s=f['source'].attrs; "
            f"expected={depth_m}; actual=float(s['z']); "
            "assert np.isclose(actual, expected), (actual, expected)"
        )
        run([str(PYTHON), "-c", verify_source], destination / "source_check.log", run_environment)
        shutil.copy2(CASE / "config.h5", destination / "config.h5")

        # Produce all three force directions with the diagnostic CUDA solver.
        for direction in "xyz":
            (CASE / "wavefields" / direction).mkdir(parents=True)
            run(
                [str(ROOT / "bin-debug/gf_solver_elastic_cuda"), "--direction", direction],
                destination / f"forward_{direction}.log",
                run_environment,
            )

        # Keep only the long-axis validation tiles.
        tile_arguments = [
            argument
            for tile_x, tile_y in ((1, 2), (2, 2), (3, 2))
            for argument in ("--tile-x", str(tile_x), "--tile-y", str(tile_y))
        ]
        run(
            [
                str(ROOT / "bin-debug/gf_postprocess"),
                "model.h5",
                "config.h5",
                "--fx",
                "wavefields/x/",
                "--fy",
                "wavefields/y/",
                "--fz",
                "wavefields/z/",
                *tile_arguments,
                "-o",
                str(destination / "greenfun"),
            ],
            destination / "postprocess.log",
            run_environment,
        )
        tiles = list((destination / "greenfun").glob("tile_*.h5"))
        if len(tiles) != 3:
            raise RuntimeError(f"Expected 3 tiles at {depth_m} m, found {len(tiles)}")
        (destination / "complete.txt").write_text("Three tiles generated.\n")

        # Remove only the full-domain wavefields after their tiles are archived.
        shutil.rmtree(CASE / "wavefields")
        print(
            f"DONE source depth={depth_m} m elapsed={time.monotonic() - case_started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
