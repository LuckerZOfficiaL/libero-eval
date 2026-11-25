#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
echo "=== [fp] START $(date) ==="
if .venv/bin/python sweep/fingerprint_pass.py; then echo "=== [fp] OK $(date) ==="; else echo "=== [fp] FAILED (exit $?) $(date) ==="; exit 1; fi
