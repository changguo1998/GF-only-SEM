#!/usr/bin/env bash
# ===========================================================================
# scripts/with_mem_limit.sh — run a command under a host RAM cap
# ===========================================================================
# Prevents runaway memory (e.g. postprocess full-replication OOM) from
# freezing/rebooting the machine. The whole process tree (mpirun + ranks)
# shares ONE cgroup limit; on overflow the kernel OOM-kills only the scope.
#
# Usage:
#   scripts/with_mem_limit.sh [LIMIT_GB] -- CMD [ARGS...]
#   LIMIT_GB omitted → $GF_MEM_LIMIT_GB (default 60); 0 = no limit.
#
# Mechanism:
#   1. systemd-run --user --scope (cgroup MemoryMax) — preferred
#   2. ulimit -v fallback (per-process virtual memory, weaker for MPI)
# ===========================================================================
set -u

LIMIT="${GF_MEM_LIMIT_GB:-60}"

# Optional numeric first argument overrides the limit
if [ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
	LIMIT="$1"
	shift
fi

if [ "${1:-}" = "--" ]; then
	shift
fi

if [ $# -eq 0 ]; then
	echo "usage: with_mem_limit.sh [LIMIT_GB] -- CMD [ARGS...]" >&2
	exit 2
fi

if [ "$LIMIT" = "0" ]; then
	exec "$@"
fi

# ── Preferred: systemd user scope (one cgroup for the whole tree) ──────────
if command -v systemd-run >/dev/null 2>&1 &&
	systemd-run --user --scope --quiet --collect true >/dev/null 2>&1; then
	echo "mem-limit: ${LIMIT}G cgroup scope for: $*" >&2
	exec systemd-run --user --scope --quiet --collect \
		-p "MemoryMax=${LIMIT}G" -p "MemorySwapMax=0" \
		"$@"
fi

# ── Fallback: ulimit -v (per-process; MPI ranks each inherit the cap) ─────
echo "mem-limit: systemd scope unavailable; ulimit -v ${LIMIT}G per process for: $*" >&2
ulimit -v $((LIMIT * 1024 * 1024))
exec "$@"
