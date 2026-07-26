#!/bin/bash
# ===========================================================================
# scripts/_run_case.sh — internal wrapper: source env + run compare.sh
# ===========================================================================
# Sources spack environment WITHOUT pipe so CPATH/CMAKE_PREFIX_PATH persist
# for cmake/make. Called by run_all_examples.sh.
# ===========================================================================
set -euo pipefail

CASE_DIR="$(cd "$1" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source project environment (no pipe!)
source "${PROJECT_ROOT}/scripts/env.sh" >/dev/null 2>&1 || true

# Export spack-set variables for cmake/make
export CPATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH PATH
export OMPI_CC OMPI_CXX PKG_CONFIG_PATH MANPATH

# Run the compare.sh
cd "${CASE_DIR}"
bash "${CASE_DIR}/compare.sh"
