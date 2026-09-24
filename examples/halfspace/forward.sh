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
# Uncomment ONE solver. The same solver runs all 3 directions
# (wavefield2vtk requires matching solver output across directions, so
# mixing different solvers per-direction is not supported).
#
# The config drives viscoelastic SLS parameters (q_mu/q_kappa/n_sls/
# f0_for_pml_hz are auto-injected into model.h5 by the preprocessor).
# With Q→∞ (elastic limit — the config default) the visco solver output
# is bit-identical to elastic, so any solver below is valid.
#
#   gf_solver_viscoelastic_*   SLS viscoelastic solver (recommended)
#   gf_solver_elastic_*        elastic solver (tolerates Q→∞ tau fields)
#   *_mpi           CPU OpenMPI (ranks from config.py:n_ranks)
#   *_cuda          CUDA single-GPU, no MPI
#   *_mpi_cuda      CUDA + MPI (multi-GPU cluster)
#
PROJECT_BIN="${PROJECT_BIN:-${GF_BIN_DIR:-${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}/bin}}"
MEMLIMIT="$(cd "${SCRIPT_DIR}/../.." && pwd)/scripts/with_mem_limit.sh"  # host RAM cap (GF_MEM_LIMIT_GB, 0=off)

# (A) VISCOELASTIC — CPU + MPI (default, Q→∞ elastic limit)
SOLVER="${PROJECT_BIN}/gf_solver_viscoelastic_mpi"

# (B) VISCOELASTIC — CUDA single GPU (no MPI)
# SOLVER="${PROJECT_BIN}/gf_solver_viscoelastic_cuda"

# (C) VISCOELASTIC — CUDA + MPI (multi-GPU cluster)
# SOLVER="${PROJECT_BIN}/gf_solver_viscoelastic_mpi_cuda"

# (D) ELASTIC — CPU + MPI
# SOLVER="${PROJECT_BIN}/gf_solver_elastic_mpi"

# (E) ELASTIC — CUDA single GPU (no MPI)
# SOLVER="${PROJECT_BIN}/gf_solver_elastic_cuda"

# (F) ELASTIC — CUDA + MPI (multi-GPU cluster)
# SOLVER="${PROJECT_BIN}/gf_solver_elastic_mpi_cuda"

# (G) Custom rank count (overrides config.py:n_ranks)
# SOLVER="${PROJECT_BIN}/gf_solver_viscoelastic_mpi"
# MPIRUN="mpirun -n 4"

# (H) Custom path / build variant
# SOLVER="${PROJECT_DIR}/build/forward/viscoelastic/gf_solver_viscoelastic_mpi"
# ──────────────────────────────────────────────────────

# ── Forward solver (3 directions) ───
echo ""
echo "=== Stage 3: Forward solver ==="
for DIR in x y z; do
	echo ""
	echo "--- direction=${DIR} ---"
	mkdir -p "${WORK_DIR}/wavefields/${DIR}"
	cd "${WORK_DIR}"

	# Single-GPU CUDA solvers (*_cuda, not *_mpi_cuda) run standalone (no MPI)
	if [[ ${SOLVER} == *_cuda && ${SOLVER} != *_mpi_cuda ]]; then
		echo "  solver: $(basename "${SOLVER}") (CUDA, no MPI)"
		"${SOLVER}" --direction "${DIR}"
	else
		echo "  solver: $(basename "${SOLVER}") (${N_RANKS} ranks)"
		"${MEMLIMIT}" -- ${MPIRUN:-mpirun} -n ${N_RANKS:-1} "${SOLVER}" --direction "${DIR}"
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
