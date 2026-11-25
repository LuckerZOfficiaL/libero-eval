#!/bin/bash
# EXP-89 stage 2: runs after the rank sweep (tmux e94) on the same machine; one orchestrator at a time
set -e; set -o pipefail
cd /home/lzhou/libero-eval
while tmux has-session -t e94 2>/dev/null || pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 120; done
echo "=== [e95] START $(date) ==="
if LS_SLICE=${LS_SLICE:-0:7} CHUNK=${CHUNK:-2} TORCHDYNAMO_DISABLE=1 .venv/bin/python sweep/local_shared_swap.py; then echo "=== [e95] ALL DONE $(date) ==="; else echo "=== [e95] FAILED (exit $?) $(date) ==="; exit 1; fi
