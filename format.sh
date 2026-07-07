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
SKIP_DIRS=(.venv build .git .pytest_cache __pycache__ .ruff_cache
           build_test build_tmp build.bak '*.egg-info' log wavefields)
SKIP_EXPR=()
for d in "${SKIP_DIRS[@]}"; do SKIP_EXPR+=(-path "*/$d" -prune -o); done
ALL_TMP=$(mktemp)
trap 'rm -rf "$ALL_TMP"' EXIT
find . "${SKIP_EXPR[@]}" -type f -print > "$ALL_TMP"

echo "plan to format $(cat "$ALL_TMP" | wc -l) files"

# per-file dispatch
fmt() {
    local p="$1"
    [ ! -f "$p" ] && return
    # echo "formatting $p ..."
    case "$(basename "$p")" in
        *.py) ruff format "$p" > /dev/null ;;
        *.md) mdformat "$p" > /dev/null ;;
        *.c | *.cpp | *.h | *.hh | *.hpp | *.cu) clang-format -i -style=file "$p" ;;
        CMakeLists.txt | *.cmake) cmake-format -i "$p" > /dev/null ;;
        *.toml) taplo format "$p" > /dev/null ;;
        *) ;;
    esac
}
export -f fmt

parallel -j 16 fmt :::: "$ALL_TMP"
