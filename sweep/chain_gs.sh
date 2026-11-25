#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
echo "=== [gs] START $(date) targets=$GT_TARGETS ==="
.venv/bin/python sweep/gt_stream.py
echo "=== [gs] stream exit $? $(date); refill pass ==="
CHUNK=${CHUNK:-7} .venv/bin/python sweep/gt_sweep.py
echo "=== [gs] EXIT $? $(date) ==="
