#!/bin/bash
# ==============
# halfspace/run.sh
# ==============
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

source "${SCRIPT_DIR}/setenv.sh"

echo "=== Halfspace Forward Solver Pipeline ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Example dir: ${EXAMPLE_DIR}"
echo "Work dir:    ${WORK_DIR}"
echo ""

# ── Clean work dir ───
cd "${WORK_DIR}"
if [ -d "${WORK_DIR}/partitions" ]; then
    rm -rf "${WORK_DIR}/partitions"
fi
if [ -d "${WORK_DIR}/wavefields" ]; then
    rm -rf "${WORK_DIR}/wavefields"
fi
if [ -d "${WORK_DIR}/log" ]; then
    rm -rf "${WORK_DIR}/log"
fi
mkdir -p "${WORK_DIR}/log"

# ── Step 1: Generate mesh ───
echo ""
echo "=== Step 1: Generate mesh ==="
python "${EXAMPLE_DIR}/mesh_gen.py"

# ── Step 2: Copy config to work dir and preprocess ───
# (preprocess reads mesh.h5 + config.py from CWD)
echo ""
echo "=== Step 2: Preprocess ==="
python -m preprocess

echo ""
echo "=== Preprocess outputs ==="
echo "mesh.h5:      $(du -sh mesh.h5 | cut -f1)"
echo "config.h5:    $(du -sh config.h5 | cut -f1)"
ls -hal "$WORK_DIR/partitions/"
echo ""
echo "log/preprocess.log:"
cat "${WORK_DIR}/log/preprocess.log" 2>/dev/null | tail -20 || true

# ── Step 3: Forward solver (3 directions) ───
for DIR in x y z; do
    echo ""
    echo "=== Step 3: Forward solver (direction=${DIR}) ==="
    mkdir -p "${WORK_DIR}/wavefields/${DIR}"
    cd "${WORK_DIR}"
    ${MPIRUN} -n ${N_RANKS} "${SOLVER}" --direction "${DIR}"
    cd "${SCRIPT_DIR}"
done

echo ""
echo "=== Forward outputs ==="
for DIR in x y z; do
    echo "wavefields/${DIR}/:"
    ls -lh "${WORK_DIR}/wavefields/${DIR}/" 2>/dev/null || echo "  (not found)"
done
echo ""
echo "Log files:"
ls -lh "${WORK_DIR}/log/" 2>/dev/null || echo "  (not found)"

# ── Summary ───
echo ""
echo "========================================"
echo "  Pipeline complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  mesh.h5                       Extended mesh with GLL geometry + PML flags"
echo "  config.h5                     Simulation parameters + STF"
echo "  partitions/partition_*.h5     Per-rank partition + exchange patterns"
echo "  wavefields/{x,y,z}/record_*.h5  Strain snapshots (3 force directions)"
echo ""
echo "Next step: Green's function extraction operates on GLL nodes directly"
echo "(no receiver positions required). See postprocess/AGENTS.md."
echo ""
echo "To inspect:"
echo "  h5dump -n mesh.h5"
echo "  h5dump -n config.h5"
