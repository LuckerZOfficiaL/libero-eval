#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
echo "=== [fp-tail] START $(date) ==="
if FP_RANGE=62-75 FP_REVERSE=1 CHUNK=6 .venv/bin/python sweep/fingerprint_pass.py; then echo "=== [fp-tail] OK $(date) ==="; else echo "=== [fp-tail] FAILED (exit $?) $(date) ==="; exit 1; fi
