#!/bin/bash
# ==============
# halfspace/run.sh
# ==============
# Master pipeline: runs mesh → preprocess → forward in current shell.
# Usage: bash run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Halfspace Forward Solver Pipeline ==="
echo ""

source "${SCRIPT_DIR}/forward.sh"
