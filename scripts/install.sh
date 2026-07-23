#!/usr/bin/env bash
# ===========================================================================
# scripts/install.sh — gf-calculation install wrapper (Linux FHS)
# ===========================================================================
#
# Builds and installs gf-calculation to a specified prefix (default: /usr/local).
#
# Usage:
#   scripts/install.sh                                     # → /usr/local
#   scripts/install.sh --prefix /opt/gf-calculation        # custom prefix
#   scripts/install.sh --prefix ~/.local --backend cpu     # user install
#
# Options:
#   --prefix PATH     Install directory (default: /usr/local)
#   --backend cpu|cuda Device backend (default: auto-detect)
#   -h, --help        Show this help
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX="/usr/local"
BACKEND="auto"

# ── Colors ────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

# ── Help ──────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 [--prefix PATH] [--backend BACKEND]

Options:
  --prefix PATH     Install directory (default: /usr/local)
  --backend cpu|cuda Device backend (default: auto-detect)
  -h, --help        Show this help

Examples:
  $0                                          # → /usr/local, auto-backend
  $0 --prefix /opt/gf-calculation             # custom prefix
  $0 --prefix ~/.local --backend cpu          # user install, CPU only
EOF
    exit 0
}

# ── Parse options ─────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage ;;
        --prefix)
            PREFIX="$2"; shift 2 ;;
        --backend)
            case "$2" in
                cpu|c) BACKEND="CPU" ;;
                cuda|gpu|g) BACKEND="CUDA" ;;
                *) echo -e "${RED}Unknown backend: $2${NC}" >&2; exit 1 ;;
            esac
            shift 2 ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}" >&2; usage ;;
        *)
            echo -e "${RED}Unexpected argument: $1${NC}" >&2; usage ;;
    esac
done

# ── Resolve backend ───────────────────────────────────────────────────────

if [ "$BACKEND" = "auto" ]; then
    if command -v nvcc &>/dev/null || [ -n "${CUDACXX:-}" ]; then
        BACKEND="CUDA"
    else
        BACKEND="CPU"
    fi
fi

# ── Source environment if available ───────────────────────────────────────

if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    source "${SCRIPT_DIR}/env.sh" 2>/dev/null || true
fi

echo -e "${GREEN}=== gf-calculation install ===${NC}"
echo "  Prefix:  ${PREFIX}"
echo "  Backend: ${BACKEND}"
echo ""

# ── Configure ─────────────────────────────────────────────────────────────

BUILD_DIR="${PROJECT_ROOT}/build-install"

if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing previous build-install directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

echo -e "${YELLOW}Configuring CMake (CMAKE_INSTALL_PREFIX=${PREFIX})...${NC}"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DGF_DEVICE_BACKEND="$BACKEND"

# ── Build ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Building...${NC}"
cmake --build "$BUILD_DIR" -j "$(nproc)"

# ── Install ───────────────────────────────────────────────────────────────

echo ""
echo -e "${YELLOW}Installing to ${PREFIX}...${NC}"
cmake --install "$BUILD_DIR"

# ── Summary ───────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}=== Install complete ===${NC}"
echo ""

if [ -d "${PREFIX}/bin" ]; then
    echo "Binaries installed to ${PREFIX}/bin/:"
    for f in "${PREFIX}"/bin/gf_* "${PREFIX}"/bin/*.sh; do
        if [ -f "$f" ]; then
            printf "  %s\n" "$(basename "$f")"
        fi
    done 2>/dev/null | sort
fi

echo ""
echo "Add ${PREFIX}/bin to your PATH:"
echo "  export PATH=\"${PREFIX}/bin:\$PATH\""
echo ""
echo "To uninstall:"
echo "  rm -f ${PREFIX}/bin/gf_* ${PREFIX}/bin/solver.sh"
echo "  rm -f ${PREFIX}/bin/build.sh ${PREFIX}/bin/env.sh"