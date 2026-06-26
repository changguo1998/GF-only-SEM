#!/bin/bash
# ==============
# halfspace/run.sh
# ==============
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
EXAMPLE_DIR="${PROJECT_DIR}/examples/halfspace"
WORK_DIR="${PROJECT_DIR}/examples/halfspace/output"

# Source Spack/MPI environment (optional — skip if MPI already in PATH)
if [ -f "${PROJECT_DIR}/scripts/env_setup.sh" ]; then
    echo "=== Sourcing environment ==="
    source "${PROJECT_DIR}/scripts/env_setup.sh" 2>/dev/null || true
fi

# MPI settings — read n_ranks from config.py
N_RANKS=$(python -c "import sys; sys.path.insert(0, '${EXAMPLE_DIR}'); import config; print(config.n_ranks)")
MPIRUN="${MPIRUN:-mpirun}"

# Check gf_solver
SOLVER="${PROJECT_DIR}/build/forward/gf_solver"
if [ ! -x "${SOLVER}" ]; then
    echo "ERROR: gf_solver not found at ${SOLVER}"
    echo "       Build with: cd ${PROJECT_DIR}/build && make gf_solver"
    exit 1
fi

echo "=== Halfspace Forward Solver Pipeline ==="
echo "Project dir: ${PROJECT_DIR}"
echo "Example dir: ${EXAMPLE_DIR}"
echo "Work dir:    ${WORK_DIR}"
echo ""

# ── Clean work dir ───
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# ── Step 1: Generate mesh ───
echo ""
echo "=== Step 1: Generate mesh ==="
python "${EXAMPLE_DIR}/mesh_gen.py"

# ── Step 2: Preprocess ───
echo ""
echo "=== Step 2: Preprocess ==="
python -m preprocess mesh.h5 "${EXAMPLE_DIR}/config.py"

echo ""
echo "=== Preprocess outputs ==="
echo "mesh.h5:      $(du -sh mesh.h5 | cut -f1)"
echo "config.h5:    $(du -sh configs/config.h5 | cut -f1)"
ls -la partitions/

# ── Step 3: Forward solver (3 directions) ───
for DIR in x y z; do
    echo ""
    echo "=== Step 3${DIR}: Forward solver (direction=${DIR}) ==="
    OUTFILE="${WORK_DIR}/wavefields/${DIR}/record_0.h5"
    if [ -f "${OUTFILE}" ]; then
        echo "  Already complete, skipping."
    else
        ${MPIRUN} -n ${N_RANKS} "${SOLVER}" \
            "${WORK_DIR}/partitions/" \
            "${WORK_DIR}/configs/config.h5" \
            "${WORK_DIR}/wavefields/${DIR}/" \
            --direction "${DIR}"
    fi
done

echo ""
echo "=== Forward outputs ==="
for DIR in x y z; do
    echo "wavefields/${DIR}/:"
    ls -lh "${WORK_DIR}/wavefields/${DIR}/" 2>/dev/null || echo "  (not found)"
done

# ── Summary ───
echo ""
echo "========================================"
echo "  Pipeline complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  mesh.h5                       Extended mesh with GLL geometry + PML flags"
echo "  configs/config.h5             Simulation parameters + STF"
echo "  partitions/partition_*.h5     Per-rank partition + exchange patterns"
echo "  wavefields/{x,y,z}/record_*.h5  Strain snapshots (3 force directions)"
echo ""
echo "Next step: Green's function extraction operates on GLL nodes directly"
echo "(no receiver positions required). See postprocess/AGENTS.md."
echo ""
echo "To inspect:"
echo "  h5dump -n mesh.h5"
echo "  h5dump -n configs/config.h5"