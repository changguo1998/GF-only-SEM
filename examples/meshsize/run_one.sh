#!/bin/bash
# run_one.sh <grid> <ranks> <cap_gb> — run ONE meshsize case in the foreground.
# compare.sh (with POSTPROCESS_RANKS / GF_MEM_LIMIT_GB), peak RSS sampler,
# then the fixed-receiver analytical comparison. Logs go to results/.
# Usage: bash run_one.sh 18 12 64
set -u
g="$1"
RANKS="$2"
CAP_GB="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/fullspace$g"

echo "======== fullspace$g pipeline (ranks=$RANKS cap=${CAP_GB}G) ========"
rm -f wavefields/x/record_*.h5 wavefields/y/record_*.h5 wavefields/z/record_*.h5 2>/dev/null
rm -rf wavefields/tile_indexes
POSTPROCESS_RANKS=$RANKS GF_SKIP_STAGE6=1 GF_MEM_LIMIT_GB=$CAP_GB bash compare.sh \
	>../results/compare.$g.log 2>&1 &
CPID=$!

python3 - "$CPID" "../results/peak.$g.txt" <<'PYEOF' &
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

pid = int(sys.argv[1]); peak = 0
while os.path.exists(f"/proc/{pid}"):
    s = tree_sum(pid)
    if s > peak:
        peak = s
    time.sleep(0.5)
with open(sys.argv[2], "w") as f:
    f.write(f"PEAK_RSS_MB={peak/1024:.1f}")
PYEOF

wait "$CPID"
RC=$?

# Tile-coverage guard: refuse stage6 on a partial tile set (postprocess OOM).
N_TILES=$(ls "$HERE/fullspace$g/greenfun"/tile_*.h5 2>/dev/null | wc -l)
if [ "$N_TILES" -ne 16 ]; then
	echo "ABORT: only $N_TILES/16 tiles written for grid $g — refusing stage6 (postprocess likely OOM-capped)"
	exit 1
fi

cd "$HERE"
"$HERE/../../.venv/bin/python" "$HERE/../../examples/_shared/analytical_compare.py" \
	"fullspace$g/greenfun/" --fullspace --fixed-receivers "$HERE/receivers_fixed.npy" \
	>"$HERE/results/stage6_fixed.$g.log" 2>&1
RC2=$?

echo "grid $g: compare.sh rc=$RC, stage6 rc=$RC2"
echo "peak: $(cat "$HERE/results/peak.$g.txt" 2>/dev/null)"
exit $((RC == 0 && RC2 == 0 ? 0 : 1))
