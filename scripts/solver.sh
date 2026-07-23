#!/usr/bin/env bash
# ===========================================================================
# scripts/solver.sh — gf-calculation solver selector and launcher
# ===========================================================================
#
# Selects the correct solver executable (elastic/viscoelastic × CPU/CUDA
# × MPI/no-MPI) and launches it with appropriate arguments.
#
# Usage:
#   scripts/solver.sh                                    # interactive menu
#   scripts/solver.sh --physics elastic --backend cpu --mpi -- --direction x
#   scripts/solver.sh --physics elastic --backend cuda -- --direction x
#
# Options:
#   --physics elastic|viscoelastic    Solver physics (required in CLI mode)
#   --backend  cpu|cuda              Device backend (required in CLI mode)
#   --mpi                             Enable MPI (default: auto-detect)
#   --n-ranks N                       Number of MPI ranks (default: from config.py)
#   -h, --help                        Show this help
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/bin"

# ── Colors ────────────────────────────────────────────────────────────────

BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ── Defaults ──────────────────────────────────────────────────────────────

PHYSICS=""
BACKEND=""
USE_MPI=""
N_RANKS=""
SOLVER_BIN=""

# ── Help ──────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
${BOLD}gf-calculation solver selector${NC}

Usage: $0 [--physics PHYSICS] [--backend BACKEND] [--mpi] [--n-ranks N] [-- SOLVER_ARGS...]

${BOLD}Options:${NC}
  --physics elastic|viscoelastic    Solver physics
  --backend  cpu|cuda              Device backend
  --mpi                             Force MPI mode (auto-detected if omitted)
  --n-ranks N                       Number of MPI ranks (default: from config.py)
  -h, --help                        Show this help

${BOLD}Examples:${NC}
  $0 --physics elastic --backend cpu --mpi -- --direction x
  $0 --physics elastic --backend cuda -- --direction x
  $0 --physics viscoelastic --backend cpu --mpi --n-ranks 16 -- --direction x
  $0                                              # interactive mode

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
    local physics="$1"
    local backend="$2"
    local mpi="$3"

    # Canonicalize
    case "$physics" in
        e|elastic)       physics="elastic" ;;
        v|visco|viscoelastic) physics="viscoelastic" ;;
    esac
    case "$backend" in
        cpu|c)  backend="cpu" ;;
        cuda|gpu|g) backend="cuda" ;;
    esac

    local name="gf_solver_${physics}"
    if [ "$backend" = "cuda" ]; then
        if [ "$mpi" = "mpi" ]; then
            name="${name}_mpi_cuda"
        else
            name="${name}_cuda"
        fi
    fi

    local path="${BIN_DIR}/${name}"
    if [ ! -x "$path" ]; then
        echo -e "${RED}ERROR: solver not found: ${path}${NC}" >&2
        echo "" >&2
        echo "Available solvers:" >&2
        find_binaries | while read -r b; do echo "  $b" >&2; done
        echo "" >&2
        echo "Build with: scripts/build.sh" >&2
        return 1
    fi
    echo "$path"
}

# ── Interactive mode ──────────────────────────────────────────────────────

interactive() {
    echo -e "${BOLD}=== gf-calculation solver selector ===${NC}"
    echo ""

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
        echo "Build with: scripts/build.sh"
        return 1
    fi

    echo "Available solvers:"
    for i in "${!bins[@]}"; do
        printf "  ${GREEN}%2d${NC}) %-50s %s\n" $((i + 1)) "$(basename "${bins[$i]}")" "${labels[$i]}"
    done
    echo ""

    local choice=""
    read -r -p "Select solver [1-${#bins[@]}]: " choice
    if [[ ! "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#bins[@]}" ]; then
        echo -e "${RED}Invalid selection${NC}"
        return 1
    fi

    SOLVER_BIN="${bins[$((choice - 1))]}"
    echo -e "Selected: ${GREEN}$(basename "$SOLVER_BIN")${NC}"
    echo ""

    # Detect from binary name
    local bin_name
    bin_name="$(basename "$SOLVER_BIN")"
    if [[ "$bin_name" == *viscoelastic* ]]; then PHYSICS="viscoelastic"; else PHYSICS="elastic"; fi
    if [[ "$bin_name" == *_cuda* ]]; then BACKEND="cuda"; else BACKEND="cpu"; fi
    USE_MPI="mpi"
    if [[ "$bin_name" == *_cuda ]] && [[ "$bin_name" != *_mpi_cuda ]]; then USE_MPI="nompi"; fi

    if [ "$USE_MPI" = "mpi" ]; then
        read -r -p "Number of MPI ranks [default: auto from config.py]: " N_RANKS
    fi

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
    # Parse options
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help) usage ;;
            --physics)
                PHYSICS="$2"; shift 2 ;;
            --backend)
                BACKEND="$2"; shift 2 ;;
            --mpi)
                USE_MPI="mpi"; shift ;;
            --n-ranks)
                N_RANKS="$2"; shift 2 ;;
            --)
                shift; break ;;
            -*)
                echo -e "${RED}Unknown option: $1${NC}" >&2; usage ;;
            *)
                echo -e "${RED}Unexpected argument: $1${NC}" >&2; usage ;;
        esac
    done
    local solver_args=("$@")

    # Interactive if no physics
    if [ -z "$PHYSICS" ]; then
        interactive || return 1
    else
        if [ -z "$BACKEND" ]; then
            echo -e "${RED}--backend is required (cpu or cuda)${NC}" >&2
            return 1
        fi
        SOLVER_BIN="$(resolve_solver "$PHYSICS" "$BACKEND" "${USE_MPI:-auto}")" || return 1
    fi

    # Validate
    if [ ! -x "$SOLVER_BIN" ]; then
        echo -e "${RED}ERROR: solver not executable: ${SOLVER_BIN}${NC}" >&2
        return 1
    fi

    # Launch
    echo -e "${BOLD}Launching:${NC} $(basename "$SOLVER_BIN") ${solver_args[*]:-}"
    echo ""

    local use_mpi=true
    if [[ "$(basename "$SOLVER_BIN")" == *_cuda ]] && [[ "$(basename "$SOLVER_BIN")" != *_mpi_cuda ]]; then
        use_mpi=false
    fi

    if $use_mpi; then
        local n="${N_RANKS:-1}"
        if [ -z "$N_RANKS" ] || [ "$n" = "1" ]; then
            local config_ranks=""
            if [ -f "config.py" ]; then
                config_ranks=$(python3 -c "import config; print(config.n_ranks)" 2>/dev/null || echo "")
            fi
            if [ -n "$config_ranks" ] && [ "$config_ranks" != "1" ]; then
                n="$config_ranks"
            fi
        fi
        echo -e "  MPI ranks: ${GREEN}${n}${NC}"
        exec ${MPIRUN:-mpirun} -n "$n" "$SOLVER_BIN" "${solver_args[@]}"
    else
        echo -e "  Mode: ${CYAN}CUDA single-GPU (no MPI)${NC}"
        exec "$SOLVER_BIN" "${solver_args[@]}"
    fi
}

main "$@"