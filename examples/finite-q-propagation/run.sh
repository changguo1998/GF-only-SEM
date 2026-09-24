#!/bin/bash
# End-to-end finite-Q attenuation and dispersion validation.
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
BACKEND="${1:-auto}"

source "${PROJECT_ROOT}/scripts/env.sh" >/dev/null 2>&1 || true
BIN_DIR="${GF_BIN_DIR:-${PROJECT_ROOT}/bin-debug}"

if [ "${BACKEND}" = "auto" ]; then
	if nvidia-smi >/dev/null 2>&1; then
		BACKEND="cuda"
	else
		BACKEND="cpu"
	fi
fi
if [ "${BACKEND}" != "cpu" ] && [ "${BACKEND}" != "cuda" ]; then
	echo "Usage: $0 [auto|cpu|cuda]" >&2
	exit 2
fi

cd "${CASE_DIR}"
rm -rf model.h5 config.h5 partitions log wavefields elastic_wavefields viscoelastic_wavefields
PYTHONPATH="${PROJECT_ROOT}" "${PROJECT_ROOT}/.venv/bin/python" mesh_gen.py
PYTHONPATH="${PROJECT_ROOT}" "${PROJECT_ROOT}/.venv/bin/python" -m preprocess

run_solver() {
	local physics="$1"
	if [ "${BACKEND}" = "cuda" ]; then
		"${BIN_DIR}/gf_solver_${physics}_cuda" --direction y
	else
		mpirun -n 2 "${BIN_DIR}/gf_solver_${physics}_mpi" --direction y
	fi
}

echo "=== Elastic baseline (${BACKEND}) ==="
run_solver elastic
mv wavefields elastic_wavefields

echo "=== Finite-Q SLS (${BACKEND}) ==="
run_solver viscoelastic
mv wavefields viscoelastic_wavefields

echo "=== Analytical attenuation and dispersion ==="
"${PROJECT_ROOT}/.venv/bin/python" \
	"${PROJECT_ROOT}/examples/_shared/finite_q_propagation.py" "${CASE_DIR}"
