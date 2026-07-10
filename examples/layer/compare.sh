#!/bin/bash
# ==============
# layer/compare.sh
# ==============
# Orchestration: generate PyFK layered reference → (future) SEM → compare.
# Usage: bash compare.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# PyFK Python from the local environment
PYFK_PYTHON="${SCRIPT_DIR}/.pyfk-venv/bin/python"
if [ ! -x "${PYFK_PYTHON}" ]; then
    echo "ERROR: PyFK not found at ${PYFK_PYTHON}"
    echo "Install: cd ${SCRIPT_DIR} && uv venv .pyfk-venv --python 3.9 && \\
    .pyfk-venv/bin/pip install pyfk obspy"
    exit 1
fi

# ── Stage L1: PyFK layered reference ───
echo ""
echo "=== Stage L1: PyFK layered reference ==="
cd "${WORK_DIR}"
"${PYFK_PYTHON}" "${WORK_DIR}/reference.py" \
  --source 0 0 490 \
  --receiver 5000 0 0 \
  --output "${WORK_DIR}/layer_reference.npz" \
  --n-time 500

echo ""
echo "=== Output ==="
ls -lh "${WORK_DIR}/layer_reference.npz"
echo ""
echo "=== Stage L1 complete ==="

# Stage L2 (future): SEM pipeline for layered model
# Stage L3 (future): compare with SEM