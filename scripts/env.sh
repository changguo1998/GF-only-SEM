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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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

_spack_load openmpi
_spack_load eigen
_spack_load cuda    # optional: uncomment for GPU builds (cuda@13.2.1)
_spack_load hdf5

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
