#!/bin/bash
# EXP-32 unified 84-pool screen on ODIN, behind extfull (EXP-29). Multi-day by design.
cd /home/lzhou/libero-eval
while true; do
  if pgrep -f "merge_orchestrat[o]r.py" >/dev/null; then sleep 120; continue; fi
  # no orchestrator => any tmpfs cache entry is orphaned from a killed run; clear it rather than
  # blocking on it. Waiting on a condition nothing can clear is a deadlock, and it cost ~5h once.
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if tmux ls 2>/dev/null | grep -qE "^extfull:"; then sleep 120; continue; fi
  sleep 20
  pgrep -f "merge_orchestrat[o]r.py" >/dev/null && continue
  break
done
.venv/bin/python sweep/unified_pool_driver.py
