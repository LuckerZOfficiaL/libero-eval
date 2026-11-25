#!/bin/bash
# Completes EXP-32: (a) x_goal_expert_4, lost to the task-name typo, now the 10th external target;
# (b) f28's two missing probes (donors 74, 75) -- the documented post-EXP-32 refill.
# Bracketed pgrep, in a script file, never on a tmux command line (09-07 trap).
cd /home/lzhou/libero-eval
while true; do
  if pgrep -f "merge_orchestrat[o]r.py" >/dev/null; then sleep 120; continue; fi
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if tmux ls 2>/dev/null | grep -qE "^sub:"; then sleep 120; continue; fi
  sleep 20
  pgrep -f "merge_orchestrat[o]r.py" >/dev/null && continue
  break
done
echo "=== [unionfix] EXP-78 clear, completing EXP-32 $(date) ==="
.venv/bin/python sweep/unified_pool_driver.py
