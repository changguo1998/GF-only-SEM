#!/usr/bin/env bash
# ===========================================================================
# scripts/env.sh — gf-calculation environment setup
# ===========================================================================
#
# Sets up Python venv, Spack packages (MPI, Eigen, HDF5), and adds project
# binaries to PATH. Source this before any solver or pipeline script.
#
# Usage:
#   source scripts/env.sh
#
# What it does:
#   1. Activate Python venv (.venv)
#   2. Load Spack packages (openmpi, eigen, hdf5)
#   3. Add project/bin to PATH
#   4. Export PROJECT_ROOT, BIN_DIR for downstream scripts
# ===========================================================================

ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ENV_SCRIPT_DIR/.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/bin"

export PROJECT_ROOT BIN_DIR

echo "=== gf-calculation environment setup ==="
echo "Project root: ${PROJECT_ROOT}"
echo ""

# ── 1. Python venv ────────────────────────────────────────────────────────

VENV="${PROJECT_ROOT}/.venv/bin/activate"
if [ -f "${VENV}" ]; then
	source "${VENV}"
	echo "[OK] Python venv: $(which python3)"
else
	echo "[WARN] Python venv not found at ${VENV}"
	echo "       Python packages may not be available."
	echo "       Create with: uv sync"
fi

# ── 2. Spack ──────────────────────────────────────────────────────────────

SPACK_SETUP="${HOME}/.spack/share/spack/setup-env.sh"
if [ -f "${SPACK_SETUP}" ]; then
	source "${SPACK_SETUP}"
	echo "[OK] Spack loaded"
else
	echo "[WARN] Spack not found at ${SPACK_SETUP}"
	echo "       C++ builds may fail."
fi

# Load Spack packages (best-effort)
_spack_load() {
	local pkg="$1"
	if spack load "$pkg" 2>/dev/null; then
		echo "[OK] spack: $pkg"
	else
		echo "[WARN] spack: $pkg — not found (build may fail)"
	fi
}

# OpenMPI@5.0.10 used by the current binaries. Keep the full hash so Spack
# does not resolve the stale /jncd4ux prefix from the previous environment.
_spack_load /jncd4uxo43ob3fzgi43roqzdypqrj7sn
_spack_load eigen
_spack_load cuda # optional: uncomment for GPU builds (cuda@13.2.1)
_spack_load hdf5

# Spack's database may be read-only on compute nodes even though the installed
# prefixes are usable. Recover the runtime paths directly from the installed
# tree so an existing build remains runnable (MPI/CUDA/HDF5).
SPACK_OPT_ROOT="${HOME}/.spack/opt/spack"
_prepend_prefix_bin() {
	local prefix="$1"
	if [ -d "${prefix}/bin" ]; then
		export PATH="${prefix}/bin:${PATH}"
	fi
	if [ -d "${prefix}/lib" ]; then
		export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
	fi
}

_runtime_prefix_from_linked_library() {
	local binary="$1"
	local library_name="$2"
	local library_path
	library_path=$(ldd "${binary}" 2>/dev/null | awk -v name="${library_name}" '$1 == name {print $3; exit}')
	if [ -n "${library_path}" ] && [ -f "${library_path}" ]; then
		cd "$(dirname "${library_path}")/.." && pwd
	fi
}

if [ -d "${SPACK_OPT_ROOT}" ]; then
	MPI_PREFIX=$(_runtime_prefix_from_linked_library "${BIN_DIR}/gf_solver_viscoelastic_mpi" libmpi.so.40)
	if [ -z "${MPI_PREFIX}" ]; then
		MPI_PREFIX=$(find "${SPACK_OPT_ROOT}" -maxdepth 5 -path '*/openmpi-*/bin/mpirun' -type f -perm -111 -print -quit 2>/dev/null | sed 's#/bin/mpirun$##')
	fi
	CUDA_PREFIX=$(find "${SPACK_OPT_ROOT}" -maxdepth 5 -path '*/cuda-*/bin/nvcc' -type f -perm -111 -print -quit 2>/dev/null | sed 's#/bin/nvcc$##')
	HDF5_PREFIX=$(_runtime_prefix_from_linked_library "${BIN_DIR}/gf_solver_viscoelastic_mpi" libhdf5.so.310)
	[ -n "${MPI_PREFIX}" ] && _prepend_prefix_bin "${MPI_PREFIX}"
	[ -n "${CUDA_PREFIX}" ] && _prepend_prefix_bin "${CUDA_PREFIX}"
	[ -n "${HDF5_PREFIX}" ] && _prepend_prefix_bin "${HDF5_PREFIX}"
	[ -n "${MPI_PREFIX}" ] && echo "[OK] runtime MPI: ${MPI_PREFIX}"
	[ -n "${CUDA_PREFIX}" ] && echo "[OK] runtime CUDA: ${CUDA_PREFIX}"
	[ -n "${HDF5_PREFIX}" ] && echo "[OK] runtime HDF5: ${HDF5_PREFIX}"
fi

# ── 3. Project binaries ───────────────────────────────────────────────────

if [ -d "$BIN_DIR" ]; then
	export PATH="${BIN_DIR}:${PATH}"
	count=$(ls -1 "$BIN_DIR"/gf_* 2>/dev/null | wc -l)
	echo "[OK] ${count} solver/tool binaries in ${BIN_DIR}"
else
	echo "[INFO] ${BIN_DIR}/ does not exist yet — build first with scripts/build.sh"
fi

echo ""
echo "=== Environment ready ==="
echo "  Solvers:  scripts/solver.sh"
echo "  Build:    scripts/build.sh"
echo ""
