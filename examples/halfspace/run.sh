#!/bin/bash
# ==============
# halfspace/run.sh
# ==============
# Master pipeline: runs all stages in sequence.
# Sources the stage scripts individually for clarity.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Halfspace Forward Solver Pipeline ==="
echo ""

bash "${SCRIPT_DIR}/mesh.sh"
echo ""
echo "────────────────────────────────────────"
echo ""

bash "${SCRIPT_DIR}/preprocess.sh"
echo ""
echo "────────────────────────────────────────"
echo ""

bash "${SCRIPT_DIR}/forward.sh"
