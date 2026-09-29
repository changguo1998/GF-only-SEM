#!/bin/bash
# End-to-end large half-space far-tile validation.
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"

source "${PROJECT_ROOT}/scripts/env.sh" >/dev/null 2>&1 || true
BIN_DIR="${GF_BIN_DIR:-${PROJECT_ROOT}/bin-debug}"

if [ "${GF_REUSE_WAVEFIELDS:-0}" != "1" ] && ! nvidia-smi >/dev/null 2>&1; then
	echo "ERROR: this large validation requires an accessible CUDA device" >&2
	exit 1
fi

mapfile -t POSTPROCESS_TILE_ARGS < <(PYTHONPATH="${CASE_DIR}" \
	"${PROJECT_ROOT}/.venv/bin/python" -c \
	'import config
for tile_x, tile_y in config.postprocess_tiles:
    print("--tile-x")
    print(tile_x)
    print("--tile-y")
    print(tile_y)')
mapfile -t EXPECTED_TILES < <(PYTHONPATH="${CASE_DIR}" \
	"${PROJECT_ROOT}/.venv/bin/python" -c \
	'import config
for tile_x, tile_y in config.postprocess_tiles:
    print(f"greenfun/tile_x{tile_x:03d}_y{tile_y:03d}.h5")')

cd "${CASE_DIR}"
check_preprocessed_source() {
	PYTHONPATH="${CASE_DIR}" "${PROJECT_ROOT}/.venv/bin/python" -c \
		'import h5py
import numpy as np
import config
expected_z = 0.0 if config.source_z_m is None else config.source_z_m
expected = np.array([config.source_x_m, config.source_y_m, expected_z])
with h5py.File("config.h5", "r") as config_file:
    source = config_file["source"].attrs
    actual = np.array([source["x"], source["y"], source["z"]])
if not np.allclose(actual, expected):
    raise SystemExit(f"ERROR: preprocessed source {actual} does not match config.py {expected}")'
}

if [ "${GF_REUSE_WAVEFIELDS:-0}" = "1" ]; then
	test -f model.h5
	test -f config.h5
	test -d partitions
	check_preprocessed_source
	for direction in x y z; do
		test -n "$(find "wavefields/${direction}" -maxdepth 1 -name 'record_*.h5' -print -quit)"
	done
	rm -rf greenfun lamb_reference.npz lamb_comparison.npz lamb_distance_depth.csv
elif [ "${GF_REUSE_PREPROCESSED:-0}" = "1" ]; then
	test -f model.h5
	test -f config.h5
	test -d partitions
	check_preprocessed_source
	rm -rf wavefields greenfun lamb_reference.npz lamb_comparison.npz lamb_distance_depth.csv
else
	rm -rf model.h5 config.h5 partitions log wavefields greenfun lamb_reference.npz lamb_comparison.npz \
		lamb_distance_depth.csv
	PYTHONPATH="${PROJECT_ROOT}" "${PROJECT_ROOT}/.venv/bin/python" mesh_gen.py
	PYTHONPATH="${PROJECT_ROOT}" "${PROJECT_ROOT}/.venv/bin/python" -m preprocess
fi

if [ "${GF_REUSE_WAVEFIELDS:-0}" != "1" ]; then
	for direction in x y z; do
		echo "=== CUDA forward: ${direction} ==="
		mkdir -p "wavefields/${direction}"
		"${BIN_DIR}/gf_solver_elastic_cuda" --direction "${direction}"
	done
fi

echo "=== Postprocess selected comparison tiles ==="
"${BIN_DIR}/gf_postprocess" model.h5 config.h5 \
	--fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
	"${POSTPROCESS_TILE_ARGS[@]}" -o greenfun/

for EXPECTED_TILE in "${EXPECTED_TILES[@]}"; do
	test -f "${EXPECTED_TILE}"
done
test "$(find greenfun -maxdepth 1 -name 'tile_*.h5' -type f | wc -l)" -eq "${#EXPECTED_TILES[@]}"

echo "=== Lamb comparison across distance and depth ==="
"${PROJECT_ROOT}/.venv/bin/python" compare_grid.py
