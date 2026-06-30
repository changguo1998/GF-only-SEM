#!/bin/bash
# ==============
# halfspace/preprocess.sh
# ==============
# Stage 2: mesh + preprocess (GLL geometry, materials, partition).
# Usage: source preprocess.sh   (or bash preprocess.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# Mesh stage first (sources setenv.sh + defines helpers)
source "${SCRIPT_DIR}/mesh.sh"

# ── Preprocess ───
echo ""
echo "=== Stage 2: Preprocess ==="
echo "--- python -m preprocess ---"
python -m preprocess

echo ""
echo "=== Preprocess outputs ==="
echo "model.h5:      $(du -sh model.h5 | cut -f1)"
echo "config.h5:    $(du -sh config.h5 | cut -f1)"
showdir "${WORK_DIR}/partitions/"
echo ""
echo "log/preprocess.log:"
cat "${WORK_DIR}/log/preprocess.log" 2>/dev/null | tail -20 || true
echo ""
echo "=== VTK output ==="
cd "${WORK_DIR}"
echo "--- model2vtk ---"
python -m tools.model2vtk
echo ""
echo "--- partition2vtk ---"
python -m tools.partition2vtk
echo ""
showdir "${WORK_DIR}/vtk/"
echo ""

echo ""
echo "=== Stage 2 complete ==="
