#!/bin/bash
# after e95b on shiva: refill the 5 errored e94 cells (targets 50, 15, 56 = TARGETS[2:5]) and confirm them
set -o pipefail
cd /home/lzhou/libero-eval
while tmux has-session -t e95b 2>/dev/null || pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
RS_SLICE=2:5 CHUNK=6 .venv/bin/python sweep/rank_sweep.py && RS_SLICE=2:5 CHUNK=6 .venv/bin/python sweep/rank_sweep.py --confirm
echo "=== [e94fill] EXIT $? $(date) ==="
