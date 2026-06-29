#!/bin/bash
# ==============
# halfspace/lib.sh
# ==============
# Shared helper functions for stage scripts.
# Source this file before using helper functions.

showdir() {
    local p
    p=${1:-'.'}
    { ls -lh "$p" 2>/dev/null || echo "  (not found)"; } | head -n 5
    if [[ "$(ls -lvh "$p" 2>/dev/null | wc -l)" -gt 5 ]]; then
        echo "  ..."
    fi
}

clean_workdir() {
    local wd="${1:-.}"
    cd "${wd}"
    rm -f "${wd}"/*.h5
    rm -rf "${wd}/partitions" "${wd}/wavefields" "${wd}/log"
    mkdir -p "${wd}/log"
}
