#!/bin/bash
# Continuous exploratory queue for shiva (user instruction 2026-09-08: keep it busy, next meeting ~09-15).
# Runs experiments in order, each idempotent and resumable, so a restart re-enters where it stopped and
# a landing never leaves the machine idle. Add new stages at the end; finished ones skip themselves.
cd /home/lzhou/libero-eval
run() {  # run <name> <script...>
  local name=$1; shift
  echo "=== [queue] START $name $(date '+%F %T') ==="
  while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if .venv/bin/python "$@"; then echo "=== [queue] OK $name $(date '+%F %T') ==="
  else echo "=== [queue] FAILED $name $(date '+%F %T') ==="; fi
}
run exp49_probe  sweep/shared_donor.py --stage probe
run exp49_sigs   sweep/shared_donor.py --stage sigs
run exp51_cheap  sweep/cheap_probe.py
run exp52_npick  sweep/npick.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
