#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
export CHUNK=6
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
rm -f results/analysis/decomp_adapters/*.safetensors 2>/dev/null || true
echo "=== [sp] START $(date) ==="
if .venv/bin/python sweep/state_probe_pass.py; then echo "=== [sp] OK $(date) ==="; else echo "=== [sp] FAILED (exit $?) $(date) ==="; exit 1; fi
