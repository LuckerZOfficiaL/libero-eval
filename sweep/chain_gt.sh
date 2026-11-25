#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true      # orphaned merged checkpoints shrink the derived worker count
echo "=== [gt] START $(date) targets=$GT_TARGETS ==="
.venv/bin/python sweep/gt_sweep.py
echo "=== [gt] EXIT $? $(date) ==="
