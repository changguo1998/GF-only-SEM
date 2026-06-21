#!/bin/bash
#=============================================================================
# Environment initialization script for gf-calculation
# Loads Spack-managed packages: CUDA-enabled OpenMPI, Eigen, HDF5
#
# Usage:
#   source scripts/env_setup.sh
#=============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== gf-calculation environment initialization ==="
echo "Project root: ${PROJECT_DIR}"
echo ""

# ---- 1. Source spack ----
SPACK_SETUP="${HOME}/.spack/share/spack/setup-env.sh"
if [ -f "${SPACK_SETUP}" ]; then
    source "${SPACK_SETUP}"
    echo "[OK] Spack environment loaded from ${SPACK_SETUP}"
else
    echo "[FAIL] Spack setup not found at ${SPACK_SETUP}"
    return 1 2>/dev/null || exit 1
fi

# ---- 2. Load packages ----
# Use hash to resolve duplicate openmpi: we want the CUDA-enabled build
# /jncd4ux is openmpi@5.0.10+cuda cuda_arch=120
OPENMPI_HASH="/jncd4ux"

echo "Loading spack packages..."

if spack load ${OPENMPI_HASH}; then
    echo "[OK] openmpi@5.0.10+cuda (CUDA support enabled)"
else
    echo "[FAIL] Failed to load openmpi"
    return 1 2>/dev/null || exit 1
fi

if spack load eigen; then
    echo "[OK] eigen@3.4.0"
else
    echo "[FAIL] Failed to load eigen"
    return 1 2>/dev/null || exit 1
fi

if spack load hdf5; then
    echo "[OK] hdf5@1.14.6+mpi (Spack-managed, MPI-aware)"
else
    echo "[FAIL] Failed to load hdf5"
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "=== Environment ready ==="
echo "  CC:   $(which mpicc 2>/dev/null || echo 'not found')"
echo "  CXX:  $(which mpicxx 2>/dev/null || echo 'not found')"
echo "  HDF5: $(which h5cc 2>/dev/null || echo 'not found')"

# Check CUDA support in OpenMPI
if ompi_info --all 2>/dev/null | grep -q "opal_built_with_cuda_support.*true"; then
    echo "  CUDA: OpenMPI built with CUDA GPU buffer support"
fi