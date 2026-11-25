#!/bin/bash
# EXP-78 subset probing on ODIN, chained behind EXP-77 (tmux s16draw).
# The pgrep pattern is bracketed AND lives in a script file, never on a tmux command line (09-07 trap).
cd /home/lzhou/libero-eval
while true; do
  if pgrep -f "merge_orchestrat[o]r.py" >/dev/null; then sleep 120; continue; fi
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if tmux ls 2>/dev/null | grep -qE "^s16draw:"; then sleep 120; continue; fi
  sleep 20
  pgrep -f "merge_orchestrat[o]r.py" >/dev/null && continue
  break
done
echo "=== [chain_sub] EXP-77 clear, starting EXP-78 $(date) ==="
.venv/bin/python sweep/subset_probe.py
