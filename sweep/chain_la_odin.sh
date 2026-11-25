#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
echo "=== [la-odin] EXP-93 0:7 START $(date) ==="
LA_ARMS=b75,b125,b150 LA_SLICE=0:7 CHUNK=2 TORCHDYNAMO_DISABLE=1 .venv/bin/python sweep/la_operator_arms.py
echo "=== [la-odin] EXIT $? $(date) ==="
