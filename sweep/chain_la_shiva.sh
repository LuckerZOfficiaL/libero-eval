#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
while tmux has-session -t e94fill 2>/dev/null || pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
echo "=== [la-shiva] EXP-92 all START $(date) ==="
LA_ARMS=std,in2,top4 CHUNK=6 .venv/bin/python sweep/la_operator_arms.py
echo "=== [la-shiva] EXP-92 EXIT $? $(date) ==="
LA_ARMS=b75,b125,b150 LA_SLICE=7: CHUNK=6 .venv/bin/python sweep/la_operator_arms.py
echo "=== [la-shiva] EXIT $? $(date) ==="
