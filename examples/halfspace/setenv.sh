#!/bin/bash
# ==============
# halfspace/setenv.sh
# ==============
# Source this file before manual pipeline steps.
# Sets up PROJECT_DIR, EXAMPLE_DIR, MPI env, N_RANKS, MPIRUN, SOLVER.
#
# Usage:
#   source examples/halfspace/setenv.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXAMPLE_DIR="${PROJECT_DIR}/examples/halfspace"

# Source Spack/MPI environment (optional)
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
    return 1 2>/dev/null || exit 1
fi

export PROJECT_DIR EXAMPLE_DIR N_RANKS MPIRUN SOLVER
echo "Environment ready: N_RANKS=${N_RANKS}, MPIRUN=${MPIRUN}, SOLVER=${SOLVER}"