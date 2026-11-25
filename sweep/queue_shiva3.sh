#!/bin/bash
# Continuous queue v3 (09-08). EXP-58 inserted after EXP-57: EXP-55 showed the pair signal is real and
# reproducible (rho 0.58), so features are the bottleneck -- and route 1's features were computed on a
# bank spanning 90 OTHER scenes. Recomputing them on each target's own demo states is the most direct
# fix available and is the best-motivated untried feature we have.
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
run exp58_tbank  sweep/merged_sigs.py --targets weak --bank target --device cuda:0
run exp51_cheap  sweep/cheap_probe.py
run exp52_npick  sweep/npick.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
