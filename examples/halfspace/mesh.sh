#!/bin/bash
# ==============
# halfspace/mesh.sh
# ==============
# Stage 1: generate mesh.
# Usage: source mesh.sh   (or bash mesh.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

source "${SCRIPT_DIR}/setenv.sh"
source "${SCRIPT_DIR}/lib.sh"

echo "=== Stage 1: Generate mesh ==="
cd "${WORK_DIR}"
clean_workdir "${WORK_DIR}"

echo "--- mesh_gen.py ---"
python "${EXAMPLE_DIR}/mesh_gen.py"

echo ""
echo "mesh.h5:      $(du -sh mesh.h5 | cut -f1)"
echo ""
echo "=== Stage 1 complete ==="
