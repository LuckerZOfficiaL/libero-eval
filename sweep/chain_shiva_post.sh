#!/bin/bash
# shiva, after the e94 (0:7) chain: refill failed e94 cells of targets 7: (+ their confirm), then EXP-89 (e95) 7:
set -e; set -o pipefail
cd /home/lzhou/libero-eval
while tmux has-session -t e94 2>/dev/null || pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 120; done
echo "=== [post] e94 refill 7: START $(date) ==="
RS_SLICE=7: CHUNK=6 .venv/bin/python sweep/rank_sweep.py && RS_SLICE=7: CHUNK=6 .venv/bin/python sweep/rank_sweep.py --confirm && echo "=== [post] e94 refill OK $(date) ===" || { echo "=== [post] e94 refill FAILED $(date) ==="; exit 1; }
if LS_SLICE=7: CHUNK=6 .venv/bin/python sweep/local_shared_swap.py; then echo "=== [e95] ALL DONE $(date) ==="; else echo "=== [e95] FAILED (exit $?) $(date) ==="; exit 1; fi
