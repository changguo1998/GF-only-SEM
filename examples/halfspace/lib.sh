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

print_summary() {
    echo ""
    echo "========================================"
    echo "  Pipeline complete!"
    echo "========================================"
    echo ""
    echo "Output files:"
    echo "  mesh.h5                       Extended mesh with GLL geometry + PML flags"
    echo "  config.h5                     Simulation parameters + STF"
    echo "  partitions/partition_*.h5     Per-rank partition + exchange patterns"
    echo "  wavefields/{x,y,z}/record_*.h5  Strain snapshots (3 force directions)"
    echo ""
    echo "Next step: Green's function extraction operates on GLL nodes directly"
    echo "(no receiver positions required). See postprocess/AGENTS.md."
    echo ""
    echo "To inspect:"
    echo "  h5dump -n mesh.h5"
    echo "  h5dump -n config.h5"
}
