#!/bin/bash
# EXP-81 OOD robustness on ODIN (moved off shiva 09-16: it never needed shiva, only a free GPU pair).
# set -e + per-stage markers; CHUNK=3 caps peak builds at ~5GB (builds are 1.66GB each).
set -e
cd /home/lzhou/libero-eval
export CHUNK=1   # shared disk ~100% full: one build in flight at a time (~1.7GB)
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 30; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
rm -f results/analysis/decomp_adapters/*.safetensors 2>/dev/null || true
echo "=== [ood] START EXP-81 $(date) ==="
if .venv/bin/python sweep/ood_robust.py; then echo "=== [ood] OK EXP-81 $(date) ==="
else echo "=== [ood] FAILED EXP-81 (exit $?) $(date) ==="; exit 1; fi
