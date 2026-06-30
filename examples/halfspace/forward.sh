#!/bin/bash
# ==============
# halfspace/forward.sh
# ==============
# Stage 3: mesh + preprocess + forward solver (3 directions).
# Usage: source forward.sh   (or bash forward.sh)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"

# Prior stages first (chain sources mesh.sh → helpers + setenv)
source "${SCRIPT_DIR}/preprocess.sh"

# ── Solver variant per direction ───
# Each source direction uses a different solver backend to validate all 3.
# x: CPU + MPI (default, 16 ranks)
# y: CUDA single-GPU, no MPI
# z: CUDA + MPI (uses available GPUs)
PROJECT_BIN="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}/bin"
SOLVER_X="${PROJECT_BIN}/gf_solver_mpi"
SOLVER_Y="${PROJECT_BIN}/gf_solver_cuda"
SOLVER_Z="${PROJECT_BIN}/gf_solver_mpi_cuda"

# ── Forward solver (3 directions) ───
echo ""
echo "=== Stage 3: Forward solver ==="
for DIR in x y z; do
    echo ""
    echo "--- direction=${DIR} ---"
    mkdir -p "${WORK_DIR}/wavefields/${DIR}"
    cd "${WORK_DIR}"
    case "${DIR}" in
        x)
            echo "  solver: gf_solver_mpi (CPU+MPI, ${N_RANKS} ranks)"
            ${MPIRUN} -n ${N_RANKS} "${SOLVER_X}" --direction "${DIR}"
            ;;
        y)
            echo "  solver: gf_solver_cuda (CUDA, no MPI)"
            "${SOLVER_Y}" --direction "${DIR}"
            ;;
        z)
            echo "  solver: gf_solver_mpi_cuda (CUDA+MPI)"
            ${MPIRUN} -n ${N_RANKS} "${SOLVER_Z}" --direction "${DIR}"
            ;;
    esac
    cd "${SCRIPT_DIR}"
done
echo ""
# wavefield2vtk requires all 3 directions to have same rank structure.
# With mixed solvers (MPI+CPU, CUDA-nompi, CUDA+MPI) the record file
# distribution differs, so we skip the combined VTK conversion.
echo "--- wavefield2vtk skipped (mixed solver backends) ---"
echo "  Run per-direction: python -m tools.wavefield2vtk --verbose"
echo "  (only after clearing unmatched wavefield dirs)"
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
