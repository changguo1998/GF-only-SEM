#!/bin/bash
# ==============
# halfspace/preprocess.sh
# ==============
# Stage: generate mesh + preprocess (GLL geometry, materials, partition).
# Run standalone:  bash preprocess.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

source "${SCRIPT_DIR}/setenv.sh"
source "${SCRIPT_DIR}/lib.sh"

echo "=== Stage: Generate mesh + Preprocess ==="
cd "${WORK_DIR}"
clean_workdir "${WORK_DIR}"

# ── Mesh ───
echo "--- mesh_gen.py ---"
python "${EXAMPLE_DIR}/mesh_gen.py"

# ── Preprocess ───
echo ""
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
echo "=== Stage complete (preprocess) ==="
