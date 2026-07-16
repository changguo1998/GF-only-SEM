#!/bin/bash
# ==============
# layer/compare.sh
# ==============
# Orchestration: SEM Green's functions → PyFK layered reference → compare.
# Usage: bash compare.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# PyFK Python from the local environment
PYFK_PYTHON="${SCRIPT_DIR}/.pyfk-venv/bin/python"
if [ ! -x "${PYFK_PYTHON}" ]; then
	echo "ERROR: PyFK not found at ${PYFK_PYTHON}"
	echo "Install: cd ${SCRIPT_DIR} && uv venv .pyfk-venv --python 3.9 && .pyfk-venv/bin/pip install pyfk obspy"
	exit 1
fi

# ── Stage L1: SEM pipeline — mesh → preprocess → forward → postprocess ───
echo ""
echo "=== Stage L1: SEM Green's functions ==="
cd "${WORK_DIR}"
source "${SCRIPT_DIR}/postprocess.sh"

# ── Stage L2: PyFK layered reference ───
echo ""
echo "=== Stage L2: PyFK layered reference ==="
cd "${WORK_DIR}"
"${PYFK_PYTHON}" "${WORK_DIR}/reference.py" \
	"${WORK_DIR}/greenfun" \
	--source 5500 5000 0 \
	--receiver 5278 5278 250 \
	--output "${WORK_DIR}/layer_reference.npz"

# ── Stage L3: Compare ───
echo ""
echo "=== Stage L3: Compare with SEM ==="
cd "${WORK_DIR}"
python "${SCRIPT_DIR}/compare.py" \
	"${WORK_DIR}/greenfun" \
	--source 5500 5000 0 \
	--receiver 5278 5278 250 \
	--reference "${WORK_DIR}/layer_reference.npz" \
	--output "${WORK_DIR}/layer_comparison.npz" \
	--fit-scale

echo ""
echo "=== All stages complete ==="
ls -lh "${WORK_DIR}/layer_reference.npz" "${WORK_DIR}/layer_comparison.npz"
