#!/bin/bash
# Continuous queue v6. EXP-60 rewritten (stable job ids; a fold that handles S16-init variants):
# probes then signatures, then the operator arms.
cd /home/lzhou/libero-eval
run() {
  local name=$1; shift
  echo "=== [queue] START $name $(date '+%F %T') ==="
  while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if .venv/bin/python "$@"; then echo "=== [queue] OK $name $(date '+%F %T') ==="
  else echo "=== [queue] FAILED $name $(date '+%F %T') ==="; fi
}
run exp60_probes2 sweep/confirm_run.py --stage probes
run exp60_sigs    sweep/confirm_run.py --stage sigs --device cuda:0
run exp51_cheap   sweep/cheap_probe.py
run exp52_npick   sweep/npick.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
