#!/bin/bash
# EXP-28 on ODIN, strictly after exttgt. Guards against the collision that killed the first attempt:
# a queued chain can fire the instant an orchestrator exits, so re-check immediately before launching
# and require the tmpfs cache to have drained (a busy cache means someone else is mid-run).
cd /home/lzhou/libero-eval
while true; do
  if pgrep -f "merge_orchestrat[o]r.py" >/dev/null; then sleep 90; continue; fi
  if tmux ls 2>/dev/null | grep -qE "^exttgt:|^ext:|^normcheck:"; then sleep 90; continue; fi
  used=$(df -BG --output=used /dev/shm | tail -1 | tr -dc '0-9')
  if [ "${used:-0}" -gt 8 ]; then sleep 90; continue; fi
  sleep 20
  pgrep -f "merge_orchestrat[o]r.py" >/dev/null && continue   # someone grabbed it during the pause
  break
done
.venv/bin/python sweep/mixed_pool_driver.py
