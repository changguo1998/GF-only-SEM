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

# ── Helpers (available to downstream scripts via source chain) ──

showdir() {
    local p
    p=${1:-'.'}
    { ls -lh "$p" 2>/dev/null || echo "  (not found)"; } | head -n 5
    if [[ "$(ls -lvh "$p" 2>/dev/null | wc -l)" -gt 5 ]]; then
        echo "  ..."
    fi
}

clean_workdir() {
    local wd="${1:-.}"
    cd "${wd}"
    rm -f "${wd}"/*.h5
    rm -rf "${wd}/partitions" "${wd}/wavefields" "${wd}/log"
    mkdir -p "${wd}/log"
}

# ── Stage 1 ──

echo "=== Stage 1: Generate mesh ==="
cd "${WORK_DIR}"
clean_workdir "${WORK_DIR}"

echo "--- mesh_gen.py ---"
python "${EXAMPLE_DIR}/mesh_gen.py"

echo ""
echo "mesh.h5:      $(du -sh mesh.h5 | cut -f1)"
echo ""
echo "=== Stage 1 complete ==="
