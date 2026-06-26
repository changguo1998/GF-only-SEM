#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$ROOT"
export ROOT

# load env
source "$ROOT/.venv/bin/activate"
source "$HOME/.spack/share/spack/setup-env.sh"
spack load llvm

# find target files
SKIP_DIRS=(.venv build .git .pytest_cache __pycache__)
SKIP_EXPR=()
for d in "${SKIP_DIRS[@]}"; do SKIP_EXPR+=(-not -path "*/$d/*"); done
ALL_TMP=$(mktemp)
trap 'rm -rf "$ALL_TMP"' EXIT
find . -type f "${SKIP_EXPR[@]}" >"$ALL_TMP"

# per-file dispatch
fmt() {
    local p="$1"
    [ ! -f "$p" ] && return
    case "$(basename "$p")" in
    *.py) ruff format "$p" >/dev/null ;;
    *.md) mdformat "$p" >/dev/null ;;
    *.c | *.cpp | *.h | *.hpp | *.cu) clang-format -i -style=file "$p" ;;
    esac
}
export -f fmt

parallel -j 16 fmt :::: "$ALL_TMP"
