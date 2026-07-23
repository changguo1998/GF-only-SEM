#!/usr/bin/env bash
# ===========================================================================
# scripts/install.sh — gf-calculation install wrapper
# ===========================================================================
#
# Builds and installs gf-calculation to a specified prefix (default: /usr/local).
# Follows Linux FHS conventions — binaries go to $PREFIX/bin.
#
# Usage:
#   scripts/install.sh [PREFIX] [BACKEND]
#
#   PREFIX   Install directory (default: /usr/local)
#   BACKEND  cpu | cuda (default: auto-detect)
#
# Examples:
#   scripts/install.sh                          # → /usr/local, auto-backend
#   scripts/install.sh /opt/gf-calculation cpu  # → /opt/gf-calculation, CPU only
#   scripts/install.sh ~/.local                 # → ~/.local/bin (user install)
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PREFIX="${1:-/usr/local}"
BACKEND="${2:-auto}"

# ── Colors ────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

# ── Pre-flight checks ─────────────────────────────────────────────────────

# Source environment if available
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    source "${SCRIPT_DIR}/env.sh" 2>/dev/null || true
fi

# Resolve backend
case "$BACKEND" in
    auto)
        if command -v nvcc &>/dev/null || [ -n "${CUDACXX:-}" ]; then
            BACKEND="CUDA"
        else
            BACKEND="CPU"
        fi
        ;;
    cpu|c)     BACKEND="CPU" ;;
    cuda|gpu)  BACKEND="CUDA" ;;
    *)
        echo -e "${RED}Unknown backend: $BACKEND (use cpu or cuda)${NC}" >&2
        exit 1
        ;;
esac

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

# Show what was installed
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
echo "To uninstall: rm -f ${PREFIX}/bin/gf_* ${PREFIX}/bin/solver.sh"
echo "              rm -f ${PREFIX}/bin/build.sh ${PREFIX}/bin/env.sh"