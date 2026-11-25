#!/bin/bash
cd /home/lzhou/libero-eval
run() {
  local name=$1; shift
  echo "=== [queue] START $name $(date '+%F %T') ==="
  while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
  rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
  if .venv/bin/python "$@"; then echo "=== [queue] OK $name $(date '+%F %T') ==="
  else echo "=== [queue] FAILED $name $(date '+%F %T') ==="; fi
}
run exp64_cheapext sweep/cheap_probe_ext.py
echo "=== [queue] ALL STAGES COMPLETE $(date '+%F %T') ==="
