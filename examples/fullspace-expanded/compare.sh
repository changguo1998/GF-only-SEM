#!/bin/bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CASE_DIR/../.." && pwd)"
BIN="${PROJECT_ROOT}/bin"
MEMLIMIT="${PROJECT_ROOT}/scripts/with_mem_limit.sh"
MEMORY_LIMIT_GB="${GF_MEM_LIMIT_GB:-60}"

source "${PROJECT_ROOT}/scripts/env.sh" >/dev/null 2>&1 || true

echo "=== Expanded full-space: 28 km, 22^3, 1 Hz ==="
cd "${CASE_DIR}"
PYTHONPATH="${PROJECT_ROOT}" python3 mesh_gen.py

cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" \
    -DGF_DEVICE_BACKEND=CPU \
    -DGF_USER_CONFIG="${CASE_DIR}/config_user_fullspace.cpp" \
    >/dev/null 2>&1
cmake --build "${PROJECT_ROOT}/build" --target gf_preprocess -j"$(nproc)" >/dev/null 2>&1

export PYTHONPATH="${PROJECT_ROOT}"
"${PROJECT_ROOT}/.venv/bin/python" -m preprocess
python3 "${PROJECT_ROOT}/examples/_shared/check_sem_consistency.py" "${CASE_DIR}"

rm -rf wavefields/x wavefields/y wavefields/z
rm -rf wavefields/tile_indexes
mkdir -p wavefields/x wavefields/y wavefields/z
for direction in x y z; do
    echo "=== CUDA direction=${direction} ==="
    "${MEMLIMIT}" "${MEMORY_LIMIT_GB}" -- "${BIN}/gf_solver_elastic_cuda" --direction "${direction}"
done

rm -rf greenfun
# Four ranks used 32.3 GiB for the historical 22-cube case; the shared cgroup
# limit protects this slightly larger recording region from exceeding 60 GiB.
"${MEMLIMIT}" "${MEMORY_LIMIT_GB}" -- mpirun -n "${POSTPROCESS_RANKS:-4}" \
    "${BIN}/gf_postprocess_mpi" model.h5 config.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ -o greenfun/

tile_count=$(find greenfun -maxdepth 1 -type f -name 'tile_*.h5' | wc -l)
if [ "${tile_count}" -ne 16 ]; then
    echo "FAIL: expected 16 tiles, got ${tile_count}"
    exit 1
fi

python3 "${PROJECT_ROOT}/examples/_shared/analytical_compare.py" greenfun/ --fullspace
