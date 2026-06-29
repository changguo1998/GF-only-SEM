#!/bin/bash
# ==============
# halfspace/forward.sh
# ==============
# Stage 3: mesh + preprocess + forward solver (3 directions).
# Usage: source forward.sh   (or bash forward.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# Prior stages first (chain sources mesh.sh → setenv.sh + lib.sh)
source "${SCRIPT_DIR}/preprocess.sh"

# ── Forward solver (3 directions) ───
echo ""
echo "=== Stage 3: Forward solver ==="
for DIR in x y z; do
    echo ""
    echo "--- direction=${DIR} ---"
    mkdir -p "${WORK_DIR}/wavefields/${DIR}"
    cd "${WORK_DIR}"
    ${MPIRUN} -n ${N_RANKS} "${SOLVER}" --direction "${DIR}"
    cd "${SCRIPT_DIR}"
done

echo ""
echo "=== Forward outputs ==="
for DIR in x y z; do
    echo "wavefields/${DIR}/:"
    showdir "${WORK_DIR}/wavefields/${DIR}/"
done
echo ""
echo "Log files:"
showdir "${WORK_DIR}/log/"

echo ""
echo "=== Stage 3 complete ==="
