#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
echo "=== [e80draw4] START $(date) ==="
if DRAW=4 ONLY_WEAK_FLEET=1 CHUNK=2 TORCHDYNAMO_DISABLE=1 .venv/bin/python sweep/blind_pool.py; then echo "=== [e80draw4] OK $(date) ==="; else echo "=== [e80draw4] FAILED (exit $?) $(date) ==="; exit 1; fi
