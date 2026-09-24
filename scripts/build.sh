#!/usr/bin/env bash
# ===========================================================================
# scripts/build.sh — gf-calculation build helper
# ===========================================================================
#
# Builds gf-calculation executables. Auto-detects MPI and CUDA availability.
#
# Usage:
#   scripts/build.sh                           # build all, auto-detect backend
#   scripts/build.sh --backend cpu             # CPU only
#   scripts/build.sh --backend cuda            # CPU + CUDA
#   scripts/build.sh --debug                   # diagnostic build + tests
#   scripts/build.sh --clean                   # clean selected build
#   scripts/build.sh --target gf_postprocess   # single target
#
# Options:
#   --backend cpu|cuda    Device backend (default: auto-detect)
#   --target TARGET       Build a specific target
#   --debug               Compile diagnostic code and tests
#   --clean               Remove selected build directory and rebuild
#   -j N                  Parallel jobs (default: nproc)
#   -h, --help            Show this help
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND="auto"
TARGET=""
CLEAN=false
DEBUG_MODE=OFF

# ── Colors ────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ── Help ──────────────────────────────────────────────────────────────────

usage() {
	cat <<EOF
Usage: $0 [--backend BACKEND] [--target TARGET] [--debug] [--clean] [-j N]

Options:
  --backend cpu|cuda    Device backend (default: auto-detect)
  --target TARGET       Build a specific target (e.g. gf_solver_elastic_mpi)
  --debug               Compile diagnostic code and tests (uses build-debug/)
  --clean               Remove selected build directory and rebuild
  -j N                  Parallel jobs (default: \$(nproc))
  -h, --help            Show this help

Examples:
  $0                                     # build all, auto-backend
  $0 --backend cpu                       # CPU only
  $0 --backend cuda                      # CPU + CUDA
  $0 --debug                              # diagnostic build + tests
  $0 --clean                             # clean rebuild
  $0 --target gf_postprocess             # single target
EOF
	exit 0
}

# ── Configure ─────────────────────────────────────────────────────────────

configure() {
	local backend="${1:-CPU}"
	echo -e "${YELLOW}Configuring CMake (GF_DEVICE_BACKEND=${backend})...${NC}"

	# Resolve Spack package prefixes for cmake (spack load sets env
	# but cmake find_package needs explicit hints).
	local spack_prefixes=""
	if command -v spack &>/dev/null; then
		local hdf5_root mpix_root eigen_root
		hdf5_root=$(spack location -i hdf5 2>/dev/null || echo "")
		mpix_root=$(spack location -i openmpi 2>/dev/null || echo "")
		eigen_root=$(spack location -i eigen 2>/dev/null || echo "")
		[ -n "$hdf5_root" ] && spack_prefixes="${hdf5_root};${spack_prefixes}"
		[ -n "$mpix_root" ] && spack_prefixes="${mpix_root};${spack_prefixes}"
		[ -n "$eigen_root" ] && spack_prefixes="${eigen_root};${spack_prefixes}"
	fi

	local cmake_args=(-B "$BUILD_DIR" -DGF_DEVICE_BACKEND="$backend" -DDEBUG="$DEBUG_MODE")
	if [ -n "$spack_prefixes" ]; then
		cmake_args+=(-DCMAKE_PREFIX_PATH="$spack_prefixes")
	fi

	cmake "${cmake_args[@]}" -S "$PROJECT_ROOT"
}

# ── Main ──────────────────────────────────────────────────────────────────

# Parse options
while [ $# -gt 0 ]; do
	case "$1" in
	-h | --help) usage ;;
	--backend)
		case "$2" in
		cpu | c) BACKEND="CPU" ;;
		cuda | gpu | g) BACKEND="CUDA" ;;
		*)
			echo -e "${RED}Unknown backend: $2${NC}" >&2
			exit 1
			;;
		esac
		shift 2
		;;
	--target)
		TARGET="$2"
		shift 2
		;;
	--debug)
		DEBUG_MODE=ON
		shift
		;;
	--clean)
		CLEAN=true
		shift
		;;
	-j)
		JOBS="$2"
		shift 2
		;;
	-*)
		echo -e "${RED}Unknown option: $1${NC}" >&2
		usage
		;;
	*)
		echo -e "${RED}Unexpected argument: $1${NC}" >&2
		usage
		;;
	esac
done

JOBS="${JOBS:-$(nproc)}"
if [ "$DEBUG_MODE" = "ON" ]; then
	BUILD_DIR="${PROJECT_ROOT}/build-debug"
	BIN_DIR="${PROJECT_ROOT}/bin-debug"
else
	BUILD_DIR="${PROJECT_ROOT}/build"
	BIN_DIR="${PROJECT_ROOT}/bin"
fi

if $CLEAN; then
	echo -e "${YELLOW}Cleaning build directory...${NC}"
	rm -rf "$BUILD_DIR"
fi

# Configure every invocation so DEBUG/backend changes cannot reuse stale targets.
if [ "$BACKEND" = "auto" ]; then
	if command -v nvcc &>/dev/null || [ -n "${CUDACXX:-}" ]; then
		BACKEND="CUDA"
	else
		BACKEND="CPU"
	fi
fi
configure "$BACKEND"

# Build
if [ -n "$TARGET" ]; then
	echo -e "${YELLOW}Building target: ${TARGET}${NC}"
	cmake --build "$BUILD_DIR" --target "$TARGET" -j "$JOBS"
	echo ""
	echo -e "${GREEN}Done. Binary at: ${BIN_DIR}/${TARGET}${NC}"
	echo "  Run: source scripts/env.sh"
else
	echo -e "${YELLOW}Building all targets...${NC}"
	cmake --build "$BUILD_DIR" -j "$JOBS"
	echo ""
	echo -e "${GREEN}=== Build complete ===${NC}"
	echo ""

	if [ -d "${BIN_DIR}" ]; then
		echo "Built executables:"
		ls -1 "${BIN_DIR}/" 2>/dev/null | while read -r f; do
			printf "  %s\n" "$f"
		done
	fi

	echo ""
	echo "To set up your shell:"
	echo "  source scripts/env.sh"
	echo ""
	echo "To run a solver:"
	echo "  scripts/solver.sh"
fi
