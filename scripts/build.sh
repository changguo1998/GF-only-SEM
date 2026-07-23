#!/usr/bin/env bash
# ===========================================================================
# scripts/build.sh — gf-calculation build helper
# ===========================================================================
#
# Builds gf-calculation executables. Auto-detects MPI and CUDA availability.
#
# Usage:
#   ./scripts/build.sh              # build all available targets
#   ./scripts/build.sh cpu          # CPU-only (MPI solvers)
#   ./scripts/build.sh cuda         # CPU + CUDA solvers
#   ./scripts/build.sh --clean      # clean build
#   ./scripts/build.sh -t gf_solver_elastic_mpi  # single target
#
# After building, run: source scripts/env.sh  (to add bin/ to PATH)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_MODE="${1:-all}"
TARGET="${2:-}"

# ── Colors ────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

# ── Help ──────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 [MODE] [-- TARGET]

Modes:
  (default)   Build all available targets
  cpu         CPU-only: elastic + viscoelastic MPI solvers
  cuda        CPU + CUDA: all solvers (requires CUDA toolkit)
  --clean     Remove build/ and rebuild from scratch

Options:
  -t, --target TARGET   Build a specific target (e.g. gf_solver_elastic_mpi)
  -j N                  Parallel jobs (default: \$(nproc))
  -h, --help            Show this help

Examples:
  $0                          # build everything
  $0 cpu                      # CPU only
  $0 cuda                     # CPU + CUDA
  $0 --clean                  # clean rebuild
  $0 -t gf_postprocess        # single tool
EOF
    exit 0
}

# ── Configure ─────────────────────────────────────────────────────────────

configure() {
    local backend="${1:-CPU}"
    echo -e "${YELLOW}Configuring CMake (GF_DEVICE_BACKEND=${backend})...${NC}"
    cmake -B "$BUILD_DIR" -DGF_DEVICE_BACKEND="$backend"
}

# ── Main ──────────────────────────────────────────────────────────────────

case "$BUILD_MODE" in
    -h|--help)
        usage
        ;;
    --clean)
        echo -e "${YELLOW}Cleaning build directory...${NC}"
        rm -rf "$BUILD_DIR"
        configure CPU
        ;;
    cpu)
        configure CPU
        ;;
    cuda)
        configure CUDA
        ;;
    all)
        if [ -d "$BUILD_DIR" ] && [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
            echo -e "${GREEN}Reusing existing build configuration${NC}"
        else
            # Auto-detect: try CUDA first, fall back to CPU
            if command -v nvcc &>/dev/null || [ -n "${CUDACXX:-}" ]; then
                configure CUDA
            else
                configure CPU
            fi
        fi
        ;;
    -t|--target)
        TARGET="$2"
        if [ -z "$TARGET" ]; then
            echo "ERROR: -t requires a target name"
            exit 1
        fi
        if [ ! -d "$BUILD_DIR" ]; then
            configure CPU
        fi
        echo -e "${YELLOW}Building target: ${TARGET}${NC}"
        cmake --build "$BUILD_DIR" --target "$TARGET" -j "$(nproc)"
        echo ""
        echo -e "${GREEN}Done. Binary at: bin/${TARGET}${NC}"
        echo "  Run: source scripts/env.sh"
        exit 0
        ;;
    *)
        echo "Unknown mode: $BUILD_MODE"
        usage
        ;;
esac

# ── Build ─────────────────────────────────────────────────────────────────

echo -e "${YELLOW}Building all targets...${NC}"
cmake --build "$BUILD_DIR" -j "$(nproc)"

echo ""
echo -e "${GREEN}=== Build complete ===${NC}"
echo ""

# Show what was built
if [ -d "${PROJECT_ROOT}/bin" ]; then
    echo "Built executables:"
    ls -1 "${PROJECT_ROOT}/bin/" 2>/dev/null | while read -r f; do
        printf "  %s\n" "$f"
    done
fi

echo ""
echo "To set up your shell:"
echo "  source scripts/env.sh"
echo ""
echo "To run a solver:"
echo "  scripts/solver.sh"