#!/bin/bash
# EXP-78 subset probing on shiva: 8 balanced 3-subsets probed at the COMMITTED dose vs the standard singleton
# probe, same targets, same candidate menus, same budget. Re-enabled 09-19 (gate file removed).
set -e
set -o pipefail
cd /home/lzhou/libero-eval
export CHUNK=6
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
rm -f results/analysis/decomp_adapters/*.safetensors 2>/dev/null || true
echo "=== [e78] START $(date) ==="
if .venv/bin/python sweep/subset_probe.py; then echo "=== [e78] OK $(date) ==="; else echo "=== [e78] FAILED (exit $?) $(date) ==="; exit 1; fi
