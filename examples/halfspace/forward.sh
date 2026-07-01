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

# ── Solver selection ──────────────────────────────────
# Pick ONE by uncommenting. The same solver runs all 3 directions.
# wavefield2vtk requires matching solver output format across directions,
# so mixing different solvers per-direction is not supported.
#
PROJECT_BIN="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}/bin"

# (A) CPU + MPI (default)
# SOLVER="${PROJECT_BIN}/gf_solver_mpi"

# (B) CUDA single-GPU, no MPI
SOLVER="${PROJECT_BIN}/gf_solver_cuda"
SOLVER_FLAGS="" # no MPI needed

# (C) CUDA + MPI (multi-GPU cluster)
# SOLVER="${PROJECT_BIN}/gf_solver_mpi_cuda"

# (D) MPI + CUDA with explicit rank/GPU control (override n_ranks)
# N_RANKS_CUSTOM=4
# SOLVER="${PROJECT_BIN}/gf_solver_mpi_cuda"
# MPIRUN="mpirun -n ${N_RANKS_CUSTOM}"

# (E) Custom path / build variant
# SOLVER="${PROJECT_DIR}/build/forward/gf_solver_mpi"
# ──────────────────────────────────────────────────────

# ── Forward solver (3 directions) ───
echo ""
echo "=== Stage 3: Forward solver ==="
for DIR in x y z; do
    echo ""
    echo "--- direction=${DIR} ---"
    mkdir -p "${WORK_DIR}/wavefields/${DIR}"
    cd "${WORK_DIR}"

    # gf_solver_cuda runs standalone (no MPI)
    if [[ ${SOLVER} == *gf_solver_cuda && ${SOLVER} != *mpi_cuda ]]; then
        echo "  solver: gf_solver_cuda (CUDA, no MPI)"
        "${SOLVER}" --direction "${DIR}"
    else
        echo "  solver: $(basename "${SOLVER}") (${N_RANKS} ranks)"
        ${MPIRUN:-mpirun} -n ${N_RANKS:-1} "${SOLVER}" --direction "${DIR}"
    fi

    cd "${SCRIPT_DIR}"
done

echo ""
echo "--- wavefield2vtk (cell-corner strain) ---"
cd "${WORK_DIR}"

# Prefer C++ gf_wavefield2vtk with parallel dispatch; fall back to Python
GF_WVTK="${PROJECT_BIN}/gf_wavefield2vtk"
if [ -x "${GF_WVTK}" ]; then
    # Derive n_snapshots from config.py
    N_SNAPSHOTS=$(python -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); import config; print(int(config.total_duration_s / config.output_dt_s))")
    N_PARALLEL="${OMP_NUM_THREADS:-16}"
    echo "  C++ gf_wavefield2vtk + parallel (-j${N_PARALLEL}, ${N_SNAPSHOTS} snapshots)"
    seq 0 $((N_SNAPSHOTS - 1)) | parallel -j"${N_PARALLEL}" OMP_NUM_THREADS=1 "${GF_WVTK}" --snap {}
else
    echo "  Python wavefield2vtk (sequential)"
    wavefield2vtk
fi
cd "${SCRIPT_DIR}"
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
