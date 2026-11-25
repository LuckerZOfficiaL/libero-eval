#!/bin/bash
# Continuous queue v2 (09-08). EXP-57 moved to the front: if the probe/commit state mismatch is real it
# changes how the cheap-probe and N_PICK arms should be scored, so it must be measured first.
cd /home/lzhou/libero-eval
run() {
  local name=$1; shift
  echo "=== [queue] START $name $(date '+%F %T') ==="
  while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if .venv/bin/python "$@"; then echo "=== [queue] OK $name $(date '+%F %T') ==="
  else echo "=== [queue] FAILED $name $(date '+%F %T') ==="; fi
}
run exp49_probe  sweep/shared_donor.py --stage probe
run exp49_sigs   sweep/shared_donor.py --stage sigs
run exp57_state  sweep/state_difficulty.py
run exp51_cheap  sweep/cheap_probe.py
run exp52_npick  sweep/npick.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
