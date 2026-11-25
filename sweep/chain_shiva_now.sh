#!/bin/bash
# SHIVA (free again 09-17): 1) EXP-77 remainder, the only thing that truly needs this host;
# 2) EXP-81 OOD robustness, moved off odin where a full shared disk had it crawling at CHUNK=1.
# Shiva has 117GB free and 3x49GB GPUs, so CHUNK can be generous.
set -e
cd /home/lzhou/libero-eval
export CHUNK=6
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 30; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
rm -f results/analysis/decomp_adapters/*.safetensors 2>/dev/null || true
stage () {
  echo "=== [shiva] START $1 $(date) ==="
  if .venv/bin/python "$2"; then echo "=== [shiva] OK $1 $(date) ==="
  else echo "=== [shiva] FAILED $1 (exit $?) $(date) ==="; exit 1; fi
}
stage EXP-77 sweep/s16_drawavg.py
stage EXP-81 sweep/ood_robust.py
echo "=== [shiva] ALL STAGES OK $(date) ==="
