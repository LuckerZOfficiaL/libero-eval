#!/bin/bash
# Continuous queue v4 (09-08 evening). EXP-59 first: dnorm_llm_00 is the only feature that has survived a
# donor-prior control, and the grid is the design that can confirm or kill it. Everything else waits.
cd /home/lzhou/libero-eval
run() {
  local name=$1; shift
  echo "=== [queue] START $name $(date '+%F %T') ==="
  while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if .venv/bin/python "$@"; then echo "=== [queue] OK $name $(date '+%F %T') ==="
  else echo "=== [queue] FAILED $name $(date '+%F %T') ==="; fi
}
run exp59_gridtb sweep/grid_tb_sigs.py
run exp51_cheap  sweep/cheap_probe.py
run exp52_npick  sweep/npick.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
