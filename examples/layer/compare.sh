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
PYFK_PYTHON="${SCRIPT_DIR}/.venv/bin/python"
MEMLIMIT="$(cd "${SCRIPT_DIR}/../.." && pwd)/scripts/with_mem_limit.sh"
if [ ! -x "${PYFK_PYTHON}" ]; then
	echo "ERROR: pyfk environment not found at ${PYFK_PYTHON}"
	echo "Install: cd ${SCRIPT_DIR} && uv venv .venv --python 3.9 && .venv/bin/pip install pyfk obspy h5py"
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
	--source 5778 5278 0 \
	--receiver 5278 5278 278 \
	--output "${WORK_DIR}/layer_reference.npz"

# ── Stage L3: Compare ───
echo ""
echo "=== Stage L3: Compare with SEM ==="
cd "${WORK_DIR}"
"${MEMLIMIT}" 60 -- python "${SCRIPT_DIR}/compare.py" \
	"${WORK_DIR}/greenfun" \
	--source 5778 5278 0 \
	--receiver 5278 5278 278 \
	--reference "${WORK_DIR}/layer_reference.npz" \
	--output "${WORK_DIR}/layer_comparison.npz" \
	--fit-scale

python "${SCRIPT_DIR}/../_shared/verify_waveform.py" \
	"${WORK_DIR}/layer_comparison.npz" \
	--end-time-s 2.0 \
	--min-correlation 0.98 \
	--max-fitted-rel-l2 0.20 \
	--min-scale 0.8 \
	--max-scale 1.2

echo ""
echo "=== All stages complete ==="
ls -lh "${WORK_DIR}/layer_reference.npz" "${WORK_DIR}/layer_comparison.npz"
