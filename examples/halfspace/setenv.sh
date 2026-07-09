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

# Source project env (Python venv + Spack MPI/Eigen/HDF5)
source "${PROJECT_DIR}/env_setup.sh"

# MPI settings — read n_ranks from config.py
N_RANKS=$(python -c "import sys; sys.path.insert(0, '${EXAMPLE_DIR}'); import config; print(config.n_ranks)")
MPIRUN="${MPIRUN:-mpirun}"

# Check solver binary (default: MPI+CPU). Override SOLVER for GPU variants.
SOLVER="${SOLVER:-${PROJECT_DIR}/bin/gf_solver_elastic_mpi}"
if [ ! -x "${SOLVER}" ]; then
    echo "ERROR: solver not found at ${SOLVER}"
    echo "       Build with: cd ${PROJECT_DIR}/build && cmake --build . --target gf_solver_elastic_mpi"
    return 1 2>/dev/null || exit 1
fi

# Make C++ tool binaries (gf_model2vtk, gf_postprocess, ...) available by name
export PATH="${PROJECT_DIR}/bin:${PATH}"

export PROJECT_DIR EXAMPLE_DIR N_RANKS MPIRUN SOLVER
echo "Environment ready: N_RANKS=${N_RANKS}, MPIRUN=${MPIRUN}, SOLVER=${SOLVER}"
