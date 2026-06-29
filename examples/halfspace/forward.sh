#!/bin/bash
# ==============
# halfspace/forward.sh
# ==============
# Stage: generate mesh + preprocess + forward solver (3 directions).
# Run standalone:  bash forward.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

source "${SCRIPT_DIR}/setenv.sh"
source "${SCRIPT_DIR}/lib.sh"

echo "=== Stage: Generate mesh + Preprocess + Forward ==="
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

# ── Forward solver (3 directions) ───
for DIR in x y z; do
    echo ""
    echo "=== Forward solver (direction=${DIR}) ==="
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
echo "=== Stage complete (forward) ==="
