#!/bin/bash
# EXP-81's last 3 cells, AFTER the EXP-83 evaluation finishes. One eval orchestrator per machine:
# running both at once put ~14 policy servers on 3 GPUs and killed EXP-81's session outright.
set -e
set -o pipefail
cd /home/lzhou/libero-eval
export CHUNK=3
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
sleep 30
pgrep -f "merge_orchestrat[o]r.py" >/dev/null && exit 1
echo "=== [ood] EXP-81 final cells $(date) ==="
if .venv/bin/python sweep/ood_robust.py; then echo "=== [ood] OK EXP-81 $(date) ==="
else echo "=== [ood] FAILED EXP-81 $(date) ==="; exit 1; fi
