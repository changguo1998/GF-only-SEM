#!/bin/bash
# ==============
# halfspace/compare.sh
# ==============
# Orchestration: compute SEM Green's functions → generate Lamb reference → compare.
# Usage: bash compare.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# ── Stage S1: SEM pipeline — mesh → preprocess → forward → postprocess ───
echo ""
echo "=== Stage S1: SEM Green's functions ==="
cd "${WORK_DIR}"
source "${SCRIPT_DIR}/postprocess.sh"

# ── Stage S2: Lamb analytic reference ───
echo ""
echo "=== Stage S2: Lamb analytic reference ==="
cd "${WORK_DIR}"
python "${SCRIPT_DIR}/reference.py" \
	"${WORK_DIR}/greenfun" \
	--source 5778 5278 0 \
	--receiver 5278 5278 278 \
	--output "${WORK_DIR}/lamb_reference.npz" \
	--source-depth-m 278.0

# ── Stage S3: Compare ───
echo ""
echo "=== Stage S3: Compare with SEM ==="
cd "${WORK_DIR}"
python "${SCRIPT_DIR}/compare.py" \
	"${WORK_DIR}/greenfun" \
	--source 5778 5278 0 \
	--receiver 5278 5278 278 \
	--reference "${WORK_DIR}/lamb_reference.npz" \
	--output "${WORK_DIR}/lamb_comparison.npz" \
	--fit-scale

echo ""
echo "=== All stages complete ==="
ls -lh "${WORK_DIR}/lamb_reference.npz" "${WORK_DIR}/lamb_comparison.npz"
