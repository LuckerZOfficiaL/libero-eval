#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
echo "=== [e94] START $(date) ==="
if RS_SLICE=${RS_SLICE:-0:7} CHUNK=${CHUNK:-2} TORCHDYNAMO_DISABLE=1 .venv/bin/python sweep/rank_sweep.py; then echo "=== [e94] SWEEP OK $(date) ==="; else echo "=== [e94] FAILED (exit $?) $(date) ==="; exit 1; fi
if RS_SLICE=${RS_SLICE:-0:7} CHUNK=${CHUNK:-2} TORCHDYNAMO_DISABLE=1 .venv/bin/python sweep/rank_sweep.py --confirm; then echo "=== [e94] CONFIRM OK $(date) ==="; else echo "=== [e94] CONFIRM FAILED (exit $?) $(date) ==="; exit 1; fi
echo "=== [e94] ALL DONE $(date) ==="
