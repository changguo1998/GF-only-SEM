#!/bin/bash
# Mesh-size study driver: run the fullspace18/20/22/24/28 pipelines (skip
# internal stage 6), then run the fixed-receiver comparison for all five cases.
#
# Everything is kept per-case: model.h5, config.h5, wavefields/, greenfun/,
# plus logs in examples/meshsize/results/.
set -uo pipefail

PROJ=/home/guochang/Projects/gf-calculation
MESH="$PROJ/examples/meshsize"
RES="$MESH/results"
mkdir -p "$RES"

run_case() { # $1 = 18|20|22|24|28
	local g="$1"
	local case="$MESH/fullspace$g"
	echo "======== fullspace$g pipeline ========"
	cd "$case"
	# parallelism per grid: budget = 64 GB TOTAL (hard user cap, incl. all ranks);
	# peak scales ~linearly with surviving ranks (tile-local extraction) — per-rank
	# footprint ~4.1 (18³) / 4.4 (20³) / 7.9 (22³) / 8.6 (24³) / 14 (28³) GB from
	# measured 4-rank peaks 16.5 / 17.4 / 31.7* / 34.3 / 57* GB (* = estimate).
	case "$g" in
	  18|20) RANKS=12; CAP_GB=64 ;;
	  22)    RANKS=4;  CAP_GB=64 ;;
	  24)    RANKS=4;  CAP_GB=64 ;;
	  28)    RANKS=3;  CAP_GB=64 ;;
	esac
	POSTPROCESS_RANKS=$RANKS GF_SKIP_STAGE6=1 GF_MEM_LIMIT_GB=$CAP_GB bash compare.sh >"$RES/compare.$g.log" 2>&1 &
	local cpid=$!
	python3 - "$cpid" "$RES/peak.$g.txt" <<'PYEOF' &
import os, sys, subprocess, time

def tree_sum(root_pid):
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,rss="]).decode().splitlines()
    except Exception:
        return 0
    pars, rss = {}, {}
    for ln in out:
        x = ln.split()
        if len(x) < 3:
            continue
        pars[int(x[0])] = int(x[1]); rss[int(x[0])] = int(x[2])
    tot, seen, fr = 0, set(), [root_pid]
    while fr:
        c = fr.pop()
        if c in seen:
            continue
        seen.add(c); tot += rss.get(c, 0)
        fr.extend(a for a, b in pars.items() if b == c)
    return tot

pid = int(sys.argv[1])
peak = 0
while os.path.exists(f"/proc/{pid}"):
    s = tree_sum(pid)
    if s > peak:
        peak = s
    time.sleep(0.5)
with open(sys.argv[2], "w") as f:
    f.write(f"PEAK_RSS_MB={peak/1024:.1f}")
PYEOF
	wait "$cpid"
	echo "  compare.sh rc=$?"
	stage6_fixed "$g"
}

stage6_fixed() { # $1 = 18|20|22|24|28; case dir must have greenfun/ tiles
	local g="$1"
	local case="$MESH/fullspace$g"
	echo "======== fullspace$g fixed-receiver comparison ========"
	cd "$case"
	"$PROJ/.venv/bin/python" "$PROJ/examples/_shared/analytical_compare.py" greenfun/ \
		--fullspace --fixed-receivers "$MESH/receivers_fixed.npy" \
		>"$RES/stage6_fixed.$g.log" 2>&1
	echo "  rc=$? -> $RES/stage6_fixed.$g.log"
}

echo "######## fullspace18 ########"
run_case 18
echo "######## fullspace20 ########"
run_case 20
echo "######## fullspace22 ########"
run_case 22
echo "######## fullspace24 ########"
run_case 24
run_case 28

echo "######## restore canonical fullspace-cubic (was clobbered by an earlier tmp study driver) ########"
cd "$PROJ/examples/fullspace-cubic"
POSTPROCESS_RANKS=6 GF_SKIP_STAGE6=1 GF_MEM_LIMIT_GB=64 bash compare.sh >"$RES/restore_fullspace_cubic.log" 2>&1
echo "  restore compare.sh rc=$?"

echo "ALL DONE"
