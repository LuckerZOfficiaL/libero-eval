#!/bin/bash
# Continuous queue v7 (09-09). The rollout-free direction is closed (EXP-60) and the interaction does not
# transfer between targets (EXP-61), so the remaining leverage is on the mechanism that works. EXP-62
# first: it checks whether the headline gains survive on initial states never used for selection, which
# affects every number we report.
cd /home/lzhou/libero-eval
run() {
  local name=$1; shift
  echo "=== [queue] START $name $(date '+%F %T') ==="
  while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if .venv/bin/python "$@"; then echo "=== [queue] OK $name $(date '+%F %T') ==="
  else echo "=== [queue] FAILED $name $(date '+%F %T') ==="; fi
}
run exp62_heldout sweep/heldout_states.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
