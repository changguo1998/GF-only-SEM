#!/bin/bash
# examples/fullspace-cubic/compare.sh
# fullspace-cubic — solver / backend (GPU vs CPU) comparison case
#
# Usage:  bash examples/fullspace-cubic/compare.sh
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CASE_DIR/../.." && pwd)"
BIN="${PROJECT_ROOT}/bin"
MEMLIMIT="${PROJECT_ROOT}/scripts/with_mem_limit.sh" # host RAM cap (GF_MEM_LIMIT_GB, 0=off)

# ── Environment ──────────────────────────────────────────
source "${PROJECT_ROOT}/scripts/env.sh" >/dev/null 2>&1 || true

echo "=============================================================="
echo " CASE: fullspace-cubic"
echo "=============================================================="

# ── Stage 1: Mesh ──────────────────────────────────────────
echo ""
echo "=== Stage 1: Mesh ==="
cd "${CASE_DIR}"
PYTHONPATH="${PROJECT_ROOT}" python3 mesh_gen.py

n_cell=$(python3 -c "import h5py; f=h5py.File('${CASE_DIR}/model.h5','r'); print(f['topology/cell_to_surface'].shape[0])")
n_expected=$(python3 -c "import sys; sys.path.insert(0, '${CASE_DIR}'); import config; print(config.nx_elements * config.ny_elements * config.nz_elements)")
[ "$n_cell" = "$n_expected" ] || {
	echo "FAIL: n_elements=$n_cell, expected $n_expected"
	exit 1
}
echo "  elements: $n_cell (expected $n_expected)"

# ── Stage 2: Preprocess ────────────────────────────────────
echo ""
echo "=== Stage 2: Preprocess (cpp) ==="
cd "${CASE_DIR}"
cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" \
	-DGF_DEVICE_BACKEND=CPU \
	-DGF_USER_CONFIG="${PROJECT_ROOT}/preprocess/cpp/config_user_fullspace.cpp" \
	>/dev/null 2>&1
cmake --build "${PROJECT_ROOT}/build" --target gf_preprocess -j"$(nproc)" >/dev/null 2>&1
echo "  C++ preprocessor built"

export PYTHONPATH="${PROJECT_ROOT}"
"${PROJECT_ROOT}/.venv/bin/python" -m preprocess 2>&1 | grep -E 'elements|ranks|nodes|Done|falling back' || true

[ -f "${CASE_DIR}/config.h5" ] || {
	echo "FAIL: config.h5 not created"
	exit 1
}

# ── Stage 2.5: SEM-internal parameter consistency (config.py vs artifacts) ──
echo ""
echo "=== Stage 2.5: SEM parameter consistency check ==="
python3 "${PROJECT_ROOT}/examples/_shared/check_sem_consistency.py" "${CASE_DIR}" || {
	echo "FAIL: config.py / config.h5 / model.h5 mismatch"
	exit 1
}

# ── Stage 3: Forward solver (elastic, cuda) ────────────────
echo ""
echo "=== Stage 3: Forward (elastic, cuda) ==="
echo ""
# Hermetic run: discard stale per-step records from any previous run
# (record_0_<step>.h5 — a shorter duration must not merge with longer ones).
rm -rf wavefields/x wavefields/y wavefields/z
rm -rf wavefields/tile_indexes
mkdir -p wavefields/x wavefields/y wavefields/z
cd "${CASE_DIR}"

# fullspace is the GPU vs CPU comparison case — requires a GPU.
nvidia-smi >/dev/null 2>&1 || {
	echo "SKIP: no GPU available"
	exit 0
}

GFSOLVER="${BIN}/gf_solver_elastic_cuda"
[ -x "${GFSOLVER}" ] || {
	echo "FAIL: CUDA solver not found: ${GFSOLVER}"
	exit 1
}

for DIR in x y z; do
	mkdir -p wavefields/$DIR
	echo "  direction=$DIR"
	"${MEMLIMIT}" -- "${GFSOLVER}" --direction "$DIR" 2>&1 | grep "complete" || {
		echo "FAIL: solver failed $DIR"
		exit 1
	}
done
# ── Stage 4: Postprocess ───────────────────────────────────
echo ""
echo "=== Stage 4: Postprocess ==="
echo ""
cd "${CASE_DIR}"
rm -rf greenfun # stale tiles from a previous duration must not leak through
cd "${CASE_DIR}"
"${MEMLIMIT}" -- mpirun -n "${POSTPROCESS_RANKS:-4}" "${BIN}/gf_postprocess_mpi" model.h5 config.h5 \
	--fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
	-o greenfun/ >/dev/null 2>&1 || {
	echo "FAIL: postprocess failed"
	exit 1
}

# ── Stage 5: Basic verification ────────────────────────────
echo ""
echo "=== Stage 5: Basic verification ==="
n_tiles=$(ls greenfun/tile_*.h5 2>/dev/null | wc -l)
[ "$n_tiles" -ge "4" ] || {
	echo "FAIL: n_tiles=$n_tiles < 4"
	exit 1
}
echo "  tiles: $n_tiles (expected >=4)"

python3 -c "
import h5py, numpy as np, glob
tiles = sorted(glob.glob('greenfun/tile_*.h5'))
max_disp = 0.0
for t in tiles:
    with h5py.File(t, 'r') as f:
        d = np.asarray(f['/field/displacement_tensor'])
        m = float(np.max(np.abs(d)))
        if m > max_disp: max_disp = m
print(f'max_displacement: {max_disp:.6g}')
if max_disp < 1.0:
    raise SystemExit(f'max_displacement {max_disp} too small (< 1.0)')
print('  displacement OK — PASSED')
" || {
	echo "FAIL: displacement verification failed"
	exit 1
}

# ── Stage 6: Analytical comparison (full-space Stokes) ─────
echo ""
echo "=== Stage 6: Analytical comparison (Stokes full-space) ==="
cd "${CASE_DIR}"

python3 "${PROJECT_ROOT}/examples/_shared/analytical_compare.py" greenfun/ --fullspace 2>&1 || {
	echo "FAIL: analytical comparison returned non-zero"
	echo "  (see VERIFICATION.md for detailed error analysis)"
	exit 1
}

echo ""
echo "=============================================================="
echo " CASE fullspace-cubic: COMPLETE"
echo "=============================================================="
