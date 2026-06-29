#!/bin/bash
# ==============
# halfspace/preprocess.sh
# ==============
# Stage 2: mesh + preprocess (GLL geometry, materials, partition).
# Usage: source preprocess.sh   (or bash preprocess.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# Mesh stage first (sources setenv.sh + lib.sh itself)
source "${SCRIPT_DIR}/mesh.sh"

# ── Preprocess ───
echo ""
echo "=== Stage 2: Preprocess ==="
echo "--- python -m preprocess ---"
python -m preprocess

echo ""
echo "=== Preprocess outputs ==="
echo "mesh.h5:      $(du -sh mesh.h5 | cut -f1)"
echo "config.h5:    $(du -sh config.h5 | cut -f1)"
showdir "${WORK_DIR}/partitions/"
echo ""
echo "log/preprocess.log:"
cat "${WORK_DIR}/log/preprocess.log" 2>/dev/null | tail -20 || true

echo ""
echo "=== Stage 2 complete ==="
