#!/bin/bash
# ==============
# halfspace/postprocess.sh
# ==============
# Stage 4: Green's function extraction from 3-direction strain records.
# Usage: source postprocess.sh   (or bash postprocess.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# Prior stages first
source "${SCRIPT_DIR}/forward.sh"

# ── Postprocess ───
echo ""
echo "=== Stage 4: Green's function extraction ==="
cd "${WORK_DIR}"

gf-postprocess model.h5 config.h5 \
    --fx wavefields/x/ \
    --fy wavefields/y/ \
    --fz wavefields/z/ \
    -o greenfun/

echo ""
echo "=== Green's function outputs ==="
echo "greenfun/:"
showdir "${WORK_DIR}/greenfun/"
echo ""
ls -lh "${WORK_DIR}/greenfun/" 2>/dev/null | head -20
echo ""
echo "=== Stage 4 complete ==="