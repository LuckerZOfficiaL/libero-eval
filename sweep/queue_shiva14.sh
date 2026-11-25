#!/bin/bash
cd /home/lzhou/libero-eval
run() { local n=$1; shift; echo "=== [queue] START $n $(date '+%F %T') ==="
  while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if .venv/bin/python "$@"; then echo "=== [queue] OK $n $(date '+%F %T') ==="; else echo "=== [queue] FAILED $n $(date '+%F %T') ==="; fi; }
run exp70_double sweep/double_candidates.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
