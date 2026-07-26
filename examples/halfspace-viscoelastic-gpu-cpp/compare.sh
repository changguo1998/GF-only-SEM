#!/bin/bash
# examples/halfspace-viscoelastic-gpu-cpp/compare.sh
# halfspace × viscoelastic × gpu × cpp
#
# Usage:  bash examples/halfspace-viscoelastic-gpu-cpp/compare.sh
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$CASE_DIR/../halfspace" && pwd)"
PROJECT_ROOT="$(cd "$CASE_DIR/../.." && pwd)"
BIN="${PROJECT_ROOT}/bin"

# ── Environment (recursive import) ──────────────────────────
source "${PROJECT_ROOT}/scripts/env.sh" > /dev/null 2>&1 || true
# spack load cuda  # GPU required


echo "=============================================================="
echo " CASE: halfspace-viscoelastic-gpu-cpp"
echo "=============================================================="

# ── Stage 1: Mesh ──────────────────────────────────────────
echo ""
echo "=== Stage 1: Mesh ==="
cd "${CASE_DIR}"
PYTHONPATH="${PROJECT_ROOT}" python3 mesh_gen.py

n_cell=$(python3 -c "import h5py; f=h5py.File('${CASE_DIR}/model.h5','r'); print(f['topology/cell_to_surface'].shape[0])")
[ "$n_cell" = "2916" ] || { echo "FAIL: n_elements=$n_cell, expected 2916"; exit 1; }
echo "  elements: $n_cell (expected 2916)"

# ── Stage 2: Preprocess ────────────────────────────────────
echo ""
echo "=== Stage 2: Preprocess (cpp) ==="
cd "${CASE_DIR}"
# Build C++ preprocessor with correct user config
cmake -S "${PROJECT_ROOT}" -B "${PROJECT_ROOT}/build" \
    -DGF_DEVICE_BACKEND=CPU \
    -DGF_USER_CONFIG="${PROJECT_ROOT}/preprocess/cpp/config_user_halfspace.cpp" \
    > /dev/null 2>&1
cmake --build "${PROJECT_ROOT}/build" --target gf_preprocess -j"$(nproc)" > /dev/null 2>&1
echo "  C++ preprocessor built with preprocess/cpp/config_user_halfspace.cpp"
export PYTHONPATH="${PROJECT_ROOT}"
"${PROJECT_ROOT}/.venv/bin/python" -m preprocess 2>&1 | grep -E 'elements|ranks|nodes|Done|falling back' || true

[ -f "${CASE_DIR}/config.h5" ] || { echo "FAIL: config.h5 not created"; exit 1; }

# ── Attenuation injection (SLS elastic-limit Q→∞) ──
echo ""
echo "=== Attenuation: injecting Q→∞ SLS parameters ==="
cd "${CASE_DIR}"
python3 "${PROJECT_ROOT}/examples/_shared/attenuation_inject.py" \
    "${CASE_DIR}/model.h5" \
    --q-mu 1e9 --q-kappa 1e9 --n-sls 3 --f0 2.0


# ── Stage 3: Forward solver ────────────────────────────────
echo ""
echo "=== Stage 3: Forward (viscoelastic, gpu) ==="
cd "${CASE_DIR}"
SOLVER="${BIN}/gf_solver_viscoelastic_cuda"
[ -x "${SOLVER}" ] || { echo "FAIL: solver not found: ${SOLVER}"; exit 1; }
nvidia-smi > /dev/null 2>&1 || { echo "SKIP: no GPU available"; exit 0; }

for DIR in x y z; do
    echo "  direction=$DIR"
    mkdir -p wavefields/$DIR
    "${SOLVER}" --direction "$DIR" 2>&1 | grep "complete" || { echo "FAIL: solver failed $DIR"; exit 1; }
done

# ── Stage 4: Postprocess ───────────────────────────────────
echo ""
echo "=== Stage 4: Postprocess ==="
cd "${CASE_DIR}"
"${BIN}/gf_postprocess" model.h5 config.h5 \
    --fx wavefields/x/ --fy wavefields/y/ --fz wavefields/z/ \
    -o greenfun/ 2>/dev/null || { echo "FAIL: postprocess failed"; exit 1; }

# ── Stage 5: Verify ────────────────────────────────────────
echo ""
echo "=== Stage 5: Verify ==="
n_tiles=$(ls greenfun/tile_*.h5 2>/dev/null | wc -l)
[ "$n_tiles" -ge "9" ] || { echo "FAIL: n_tiles=$n_tiles < 9"; exit 1; }
echo "  tiles: $n_tiles (expected >=9)"

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
lo = float('1.0e6')
hi = float('3.0e6')
if not (lo <= max_disp <= hi):
    raise SystemExit(f'max_displacement {max_disp} outside [{lo}, {hi}]')
print('  displacement in range — PASSED')
" || { echo "FAIL: displacement verification failed"; exit 1; }

# ── Stage 6: Elastic-limit regression ──
ELASTIC_CASE_DIR="${CASE_DIR//viscoelastic/elastic}"
ELASTIC_GREENFUN="${ELASTIC_CASE_DIR}/greenfun"
if [ -d "${ELASTIC_GREENFUN}" ]; then
    echo ""
    echo "=== Stage 6: Elastic-limit regression ==="
    cd "${CASE_DIR}"
    python3 "${PROJECT_ROOT}/examples/_shared/elastic_limit_compare.py" \
        "${ELASTIC_GREENFUN}" "${CASE_DIR}/greenfun" \
        --tol 0.01 \
        --output "${CASE_DIR}/elastic_limit_comparison.npz"
else
    echo ""
    echo "=== Stage 6: Elastic-limit regression — SKIP (no elastic reference) ==="
fi

echo ""
echo "=============================================================="
echo " CASE halfspace-viscoelastic-gpu-cpp: PASSED"
echo "=============================================================="
