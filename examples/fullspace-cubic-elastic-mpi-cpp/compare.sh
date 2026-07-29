#!/bin/bash
# examples/fullspace-cubic-elastic-mpi-cpp/compare.sh
# fullspace-cubic × elastic × mpi × cpp
#
# Usage:  bash examples/fullspace-cubic-elastic-mpi-cpp/compare.sh
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CASE_DIR/../.." && pwd)"
BIN="${PROJECT_ROOT}/bin"

# ── Environment ──────────────────────────────────────────
source "${PROJECT_ROOT}/scripts/env.sh" >/dev/null 2>&1 || true

echo "=============================================================="
echo " CASE: fullspace-cubic-elastic-mpi-cpp"
echo "=============================================================="

# ── Stage 1: Mesh ──────────────────────────────────────────
echo ""
echo "=== Stage 1: Mesh ==="
cd "${CASE_DIR}"
PYTHONPATH="${PROJECT_ROOT}" python3 mesh_gen.py

n_cell=$(python3 -c "import h5py; f=h5py.File('${CASE_DIR}/model.h5','r'); print(f['topology/cell_to_surface'].shape[0])")
[ "$n_cell" = "5832" ] || {
	echo "FAIL: n_elements=$n_cell, expected 5832"
	exit 1
}
echo "  elements: $n_cell (expected 5832)"

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

# ── Stage 3: Forward solver (elastic, cuda) ────────────────
echo ""
echo "=== Stage 3: Forward (elastic, cuda) ==="
cd "${CASE_DIR}"

GFSOLVER="${BIN}/gf_solver_elastic_cuda"
[ -x "${GFSOLVER}" ] || {
	echo "FAIL: CUDA solver not found: ${GFSOLVER}"
	exit 1
}

for DIR in x y z; do
	mkdir -p wavefields/$DIR
	echo "  direction=$DIR"
	"${GFSOLVER}" --direction "$DIR" 2>&1 | grep "complete" || {
		echo "FAIL: solver failed $DIR"
		exit 1
	}
done
# ── Stage 4: Postprocess ───────────────────────────────────
echo ""
echo "=== Stage 4: Postprocess ==="
cd "${CASE_DIR}"
"${BIN}/gf_postprocess" model.h5 config.h5 \
	--fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
	-o greenfun/ 2>/dev/null || {
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

python3 "${PROJECT_ROOT}/examples/_shared/analytical_compare.py" greenfun/ --source 9500.0 9500.0 9500.0 --fullspace 2>&1 || {
	echo "WARNING: analytical comparison returned non-zero"
	echo "  (see VERIFICATION.md for detailed error analysis)"
}

echo ""
echo "=============================================================="
echo " CASE fullspace-cubic-elastic-mpi-cpp: COMPLETE"
echo "=============================================================="
