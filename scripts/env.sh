#!/usr/bin/env bash
# ===========================================================================
# scripts/env.sh — gf-calculation environment setup
# ===========================================================================
#
# Sets up Python venv, pinned Spack development packages, and adds project
# binaries to PATH. Source this before any solver or pipeline script.
#
# Usage:
#   source scripts/env.sh
#
# What it does:
#   1. Activate Python venv (.venv)
#   2. Load the pinned Spack toolchain
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

# Package hashes verified with `spack find` on 2026-09-24. OpenMPI and HDF5
# use the CUDA 12.0 architecture stack for the local RTX 5060 Ti. Keep the
# complete hashes to avoid selecting stale package instances.
OPENMPI_HASH="zkrqzmdsne4sdpbiqv6q37xajgibtore"
CUDA_HASH="afehoi7get43pmxjnt4drc7epk7ljw2v"
HDF5_HASH="xolw2i7pcfgoqo7r7jtd25vio2aa7dhq"
EIGEN_HASH="grgpbzpilxnp6lwfevmzxadqdfw62nrh"
METIS_HASH="sgrxh75v5ajycahee73kmfqyy7ekvwxv"
LLVM_HASH="km2c7amk5eme466zsmwxrmjiigfq7vz2"

# Load Spack packages (best-effort).
_spack_load() {
	local pkg="$1"
	if spack load "$pkg" 2>/dev/null; then
		echo "[OK] spack: $pkg"
	else
		echo "[WARN] spack: $pkg — not found (build may fail)"
	fi
}

_spack_load "/${OPENMPI_HASH}"
_spack_load "/${CUDA_HASH}"
_spack_load "/${HDF5_HASH}"
_spack_load "/${EIGEN_HASH}"
_spack_load "/${METIS_HASH}"
_spack_load "/${LLVM_HASH}"

# Spack's database may be read-only on compute nodes even though the installed
# prefixes are usable. Recover the pinned prefixes directly from the installed
# tree so builds and existing binaries remain usable.
SPACK_OPT_ROOT="${HOME}/.spack/opt/spack"
_prepend_path() {
	local variable_name="$1"
	local entry="$2"
	local current_value
	local filtered_value=""
	local current_entry
	local remaining
	eval "current_value=\${${variable_name}:-}"
	remaining="${current_value}:"
	while [ -n "${remaining}" ]; do
		current_entry="${remaining%%:*}"
		remaining="${remaining#*:}"
		[ -z "${current_entry}" ] && continue
		[ "${current_entry}" = "${entry}" ] && continue
		filtered_value="${filtered_value}${filtered_value:+:}${current_entry}"
	done
	export "${variable_name}=${entry}${filtered_value:+:${filtered_value}}"
}

_remove_path_pattern() {
	local variable_name="$1"
	local path_pattern="$2"
	local current_value
	local filtered_value=""
	local current_entry
	local remaining
	eval "current_value=\${${variable_name}:-}"
	remaining="${current_value}:"
	while [ -n "${remaining}" ]; do
		current_entry="${remaining%%:*}"
		remaining="${remaining#*:}"
		[ -z "${current_entry}" ] && continue
		[[ "${current_entry}" == ${path_pattern} ]] && continue
		filtered_value="${filtered_value}${filtered_value:+:}${current_entry}"
	done
	export "${variable_name}=${filtered_value}"
}

_activate_prefix() {
	local prefix="$1"
	local library_dir
	if [ -d "${prefix}/bin" ]; then
		_prepend_path PATH "${prefix}/bin"
	fi
	for library_dir in "${prefix}/lib" "${prefix}/lib64" \
		"${prefix}/lib/x86_64-unknown-linux-gnu"; do
		[ -d "${library_dir}" ] && _prepend_path LD_LIBRARY_PATH "${library_dir}"
	done
	_prepend_path CMAKE_PREFIX_PATH "${prefix}"
}

_pinned_prefix() {
	local package_name="$1"
	local package_version="$2"
	local package_hash="$3"
	find "${SPACK_OPT_ROOT}" -mindepth 2 -maxdepth 2 -type d \
		-name "${package_name}-${package_version}-${package_hash}" -print -quit 2>/dev/null
}

if [ -d "${SPACK_OPT_ROOT}" ]; then
	OPENMPI_PREFIX=$(_pinned_prefix openmpi 5.0.10 "${OPENMPI_HASH}")
	CUDA_PREFIX=$(_pinned_prefix cuda 13.2.1 "${CUDA_HASH}")
	HDF5_PREFIX=$(_pinned_prefix hdf5 1.14.6 "${HDF5_HASH}")
	EIGEN_PREFIX=$(_pinned_prefix eigen 3.4.0 "${EIGEN_HASH}")
	METIS_PREFIX=$(_pinned_prefix metis 5.1.0 "${METIS_HASH}")
	LLVM_PREFIX=$(_pinned_prefix llvm 22.1.5 "${LLVM_HASH}")

	# Remove every OpenMPI prefix inherited from the parent shell before
	# activating the instance used by HDF5 and the project binaries.
	_remove_path_pattern PATH "${SPACK_OPT_ROOT}/*/openmpi-*/bin"
	_remove_path_pattern LD_LIBRARY_PATH "${SPACK_OPT_ROOT}/*/openmpi-*/lib"
	_remove_path_pattern CMAKE_PREFIX_PATH "${SPACK_OPT_ROOT}/*/openmpi-*"

	for prefix in "${OPENMPI_PREFIX}" "${CUDA_PREFIX}" "${HDF5_PREFIX}" "${EIGEN_PREFIX}" \
		"${METIS_PREFIX}" "${LLVM_PREFIX}"; do
		[ -n "${prefix}" ] && _activate_prefix "${prefix}"
	done

	[ -n "${CUDA_PREFIX}" ] && export CUDA_HOME="${CUDA_PREFIX}"
	[ -n "${METIS_PREFIX}" ] && export METIS_ROOT="${METIS_PREFIX}"
	[ -n "${OPENMPI_PREFIX}" ] && echo "[OK] runtime MPI: ${OPENMPI_PREFIX}"
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
