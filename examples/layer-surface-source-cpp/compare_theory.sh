#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${CASE_DIR}/../.." && pwd)"
PYFK_PYTHON="${PROJECT_ROOT}/examples/layer/.venv/bin/python"

if [ ! -x "${PYFK_PYTHON}" ]; then
	echo "ERROR: PyFK environment not found at ${PYFK_PYTHON}" >&2
	exit 1
fi

"${PYFK_PYTHON}" "${PROJECT_ROOT}/examples/layer/reference.py" "${CASE_DIR}/greenfun" \
	--source 5778 5278 0 \
	--receiver 5278 5278 0 \
	--source-depth-m 1 \
	--output "${CASE_DIR}/layer_surface_reference.npz"

"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/examples/layer/compare.py" \
	"${CASE_DIR}/greenfun" \
	--source 5778 5278 0 \
	--receiver 5278 5278 0 \
	--reference "${CASE_DIR}/layer_surface_reference.npz" \
	--output "${CASE_DIR}/layer_surface_comparison.npz" \
	--fit-scale --rebuild-index
