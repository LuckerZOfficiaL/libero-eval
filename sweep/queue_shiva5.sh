#!/bin/bash
# Continuous queue v5 (09-08 evening). EXP-60 first: a pre-registered confirmation of dnorm_llm_17 on a
# new target cohort is the only thing that can promote or kill the best rollout-free result we have.
cd /home/lzhou/libero-eval
run() {
  local name=$1; shift
  echo "=== [queue] START $name $(date '+%F %T') ==="
  while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if .venv/bin/python "$@"; then echo "=== [queue] OK $name $(date '+%F %T') ==="
  else echo "=== [queue] FAILED $name $(date '+%F %T') ==="; fi
}
run exp60_probes sweep/confirm_grid.py
run exp51_cheap  sweep/cheap_probe.py
run exp52_npick  sweep/npick.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
