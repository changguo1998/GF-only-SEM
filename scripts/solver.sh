#!/usr/bin/env bash
# ===========================================================================
# scripts/solver.sh — gf-calculation solver selector and launcher
# ===========================================================================
#
# Selects the correct solver executable (elastic / viscoelastic × CPU / CUDA
# × MPI / no-MPI) and launches it with appropriate arguments.
#
# Usage:
#   # Interactive mode (pick from menu):
#   ./scripts/solver.sh
#
#   # CLI mode:
#   ./scripts/solver.sh [PHYSICS] [BACKEND] [MPI_MODE] [-n N_RANKS] [-- ARGS...]
#
#   PHYSICS:   elastic | viscoelastic
#   BACKEND:   cpu | cuda
#   MPI_MODE:  mpi | nompi
#
# Examples:
#   ./scripts/solver.sh elastic cpu mpi -- --direction x
#   ./scripts/solver.sh elastic cpu -n 16 -- --direction x
#   ./scripts/solver.sh elastic cuda -- --direction x
#   ./scripts/solver.sh viscoelastic cpu mpi -n 8 -- --direction x
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/bin"

# Colors
BOLD='\033[1m'; RED='\033[0;31m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

# Defaults
PHYSICS=""
BACKEND=""
MPI_MODE=""
N_RANKS=""
SOLVER_BIN=""

# ── Help ──────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
${BOLD}gf-calculation solver selector${NC}

Usage: $0 [PHYSICS] [BACKEND] [MPI_MODE] [-n N] [-- SOLVER_ARGS...]

${BOLD}Positional arguments:${NC}
  PHYSICS    elastic | viscoelastic
  BACKEND    cpu | cuda
  MPI_MODE   mpi | nompi

${BOLD}Options:${NC}
  -n N       Number of MPI ranks (default: 1 for nompi, detected from config.py for mpi)
  -h, --help Show this help

${BOLD}Examples:${NC}
  $0 elastic cpu mpi -- --direction x     # elastic, CPU, MPI
  $0 elastic cuda -- --direction x         # elastic, CUDA, no MPI
  $0 elastic cpu -n 16 -- --direction x    # elastic, CPU, 16 MPI ranks
  $0 viscoelastic cpu mpi -- -d x          # viscoelastic, CPU, MPI
  $0                                       # interactive mode

${BOLD}Available binaries in ${BIN_DIR}:${NC}
EOF
    find_binaries | while read -r b; do
        local tag=""
        case "$b" in
            *elastic*cpu*|*elastic_mpi*)   tag="${GREEN}[elastic CPU+MPI]${NC}" ;;
            *elastic*cuda*mpi*)     tag="${CYAN}[elastic CUDA+MPI]${NC}" ;;
            *elastic*cuda*)         tag="${CYAN}[elastic CUDA]${NC}" ;;
            *viscoelastic*cpu*|*viscoelastic_mpi*) tag="${YELLOW}[visco CPU+MPI]${NC}" ;;
            *viscoelastic*cuda*mpi*) tag="${YELLOW}[visco CUDA+MPI]${NC}" ;;
            *viscoelastic*cuda*)     tag="${YELLOW}[visco CUDA]${NC}" ;;
        esac
        printf "  %-45s %b\n" "$b" "$tag"
    done
    echo ""
    echo "${BOLD}Environment:${NC}"
    echo "  Source scripts/env.sh first to set up Spack + Python venv."
    exit 0
}

# ── Binary discovery ──────────────────────────────────────────────────────

find_binaries() {
    if [ -d "$BIN_DIR" ]; then
        ls -1 "$BIN_DIR"/gf_solver_* 2>/dev/null || true
    fi
}

resolve_solver() {
    local physics="${1:-}"
    local backend="${2:-}"
    local mpi="${3:-}"

    # Canonicalize names
    case "$physics" in
        e|elastic)       physics="elastic" ;;
        v|visco|viscoelastic) physics="viscoelastic" ;;
    esac
    case "$backend" in
        cpu|c)           backend="cpu" ;;
        cuda|gpu|g)      backend="cuda" ;;
    esac
    case "$mpi" in
        mpi|m)           mpi="mpi" ;;
        nompi|no|n|solo) mpi="nompi" ;;
    esac

    # Determine binary name
    local name="gf_solver_${physics}"
    if [ "$backend" = "cuda" ]; then
        if [ "$mpi" = "mpi" ]; then
            name="${name}_mpi_cuda"
        else
            name="${name}_cuda"
        fi
    elif [ "$mpi" = "nompi" ]; then
        name="${name}_mpi"  # CPU always uses MPI; nompi not valid for CPU
    fi

    local path="${BIN_DIR}/${name}"
    if [ ! -x "$path" ]; then
        echo -e "${RED}ERROR: solver not found: ${path}${NC}" >&2
        echo "" >&2
        echo "Available solvers:" >&2
        find_binaries | while read -r b; do echo "  $b" >&2; done
        echo "" >&2
        echo "Build with:" >&2
        echo "  cd ${PROJECT_ROOT} && cmake -B build && cmake --build build" >&2
        return 1
    fi
    echo "$path"
}

# ── Interactive mode ──────────────────────────────────────────────────────

interactive() {
    echo -e "${BOLD}=== gf-calculation solver selector ===${NC}"
    echo ""

    # Collect available binaries
    local bins=()
    local labels=()
    while IFS= read -r b; do
        bins+=("$b")
        case "$b" in
            *elastic_mpi*)       labels+=("elastic  | CPU + MPI") ;;
            *elastic_cuda)       labels+=("elastic  | CUDA (single GPU)") ;;
            *elastic_mpi_cuda)   labels+=("elastic  | CUDA + MPI (multi-GPU)") ;;
            *viscoelastic_mpi*)  labels+=("viscoelastic | CPU + MPI") ;;
            *viscoelastic_cuda)  labels+=("viscoelastic | CUDA (single GPU)") ;;
            *viscoelastic_mpi_cuda) labels+=("viscoelastic | CUDA + MPI (multi-GPU)") ;;
            *)                   labels+=("$(basename "$b")") ;;
        esac
    done < <(find_binaries)

    if [ ${#bins[@]} -eq 0 ]; then
        echo -e "${RED}No solver binaries found in ${BIN_DIR}${NC}"
        echo "Build with: cd ${PROJECT_ROOT} && cmake -B build && cmake --build build"
        return 1
    fi

    # Display menu
    echo "Available solvers:"
    for i in "${!bins[@]}"; do
        printf "  ${GREEN}%2d${NC}) %-50s %s\n" $((i + 1)) "$(basename "${bins[$i]}")" "${labels[$i]}"
    done
    echo ""

    # Get user choice
    local choice=""
    read -r -p "Select solver [1-${#bins[@]}]: " choice
    if [[ ! "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#bins[@]}" ]; then
        echo -e "${RED}Invalid selection${NC}"
        return 1
    fi

    SOLVER_BIN="${bins[$((choice - 1))]}"
    echo -e "Selected: ${GREEN}$(basename "$SOLVER_BIN")${NC}"
    echo ""

    # Detect physics/backend/mpi from binary name
    local bin_name="$(basename "$SOLVER_BIN")"
    if [[ "$bin_name" == *viscoelastic* ]]; then
        PHYSICS="viscoelastic"
    else
        PHYSICS="elastic"
    fi
    if [[ "$bin_name" == *_cuda* ]]; then
        BACKEND="cuda"
    else
        BACKEND="cpu"
    fi
    if [[ "$bin_name" == *_mpi_cuda ]] || [[ "$bin_name" != *_cuda ]]; then
        MPI_MODE="mpi"
    else
        MPI_MODE="nompi"
    fi

    # MPI ranks
    if [ "$MPI_MODE" = "mpi" ]; then
        read -r -p "Number of MPI ranks [default: auto from config.py]: " N_RANKS
    fi

    # Extra args
    local extra=""
    read -r -p "Extra solver args (e.g. --direction x): " extra
    if [ -n "$extra" ]; then
        set -- $extra
    else
        set --
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────

main() {
    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help) usage ;;
            -n)
                N_RANKS="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                if [ -z "$PHYSICS" ]; then
                    PHYSICS="$1"
                elif [ -z "$BACKEND" ]; then
                    BACKEND="$1"
                elif [ -z "$MPI_MODE" ]; then
                    MPI_MODE="$1"
                else
                    echo -e "${RED}Unexpected argument: $1${NC}" >&2
                    usage
                fi
                shift
                ;;
        esac
    done
    local solver_args=("$@")

    # Interactive mode if no physics specified
    if [ -z "$PHYSICS" ]; then
        interactive || return 1
    else
        # Resolve calculator binary
        SOLVER_BIN="$(resolve_solver "$PHYSICS" "$BACKEND" "${MPI_MODE:-mpi}")" || return 1
    fi

    # Validate
    if [ ! -x "$SOLVER_BIN" ]; then
        echo -e "${RED}ERROR: solver not executable: ${SOLVER_BIN}${NC}" >&2
        return 1
    fi

    # Launch
    echo -e "${BOLD}Launching:${NC} $(basename "$SOLVER_BIN") ${solver_args[*]:-}"
    echo ""

    # Determine if MPI is needed
    local use_mpi=true
    if [[ "$(basename "$SOLVER_BIN")" == *_cuda ]] && [[ "$(basename "$SOLVER_BIN")" != *_mpi_cuda ]]; then
        use_mpi=false
    fi

    if $use_mpi; then
        local n="${N_RANKS:-1}"
        if [ "$n" = "1" ] || [ -z "$N_RANKS" ]; then
            # Try to read from config.py
            local config_ranks=""
            if [ -f "config.py" ]; then
                config_ranks=$(python3 -c "import config; print(config.n_ranks)" 2>/dev/null || echo "")
            fi
            if [ -n "$config_ranks" ] && [ "$config_ranks" != "1" ]; then
                n="$config_ranks"
            fi
        fi
        local mpirun="${MPIRUN:-mpirun}"
        echo -e "  MPI ranks: ${GREEN}${n}${NC}"
        exec $mpirun -n "$n" "$SOLVER_BIN" "${solver_args[@]}"
    else
        echo -e "  Mode: ${CYAN}CUDA single-GPU (no MPI)${NC}"
        exec "$SOLVER_BIN" "${solver_args[@]}"
    fi
}

main "$@"