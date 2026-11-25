#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
until grep -q "COMPLETE" results/fp_handoff.log 2>/dev/null; do sleep 120; done
n=$(ls results/merge/e92_base_f*.jsonl | wc -l); echo "=== [e85] START $(date) with $n/76 e92 files ==="
if .venv/bin/python sweep/exp85_prior.py --seeds 36 --epochs 150 --lr 1e-3 --tag full 2>&1 | grep --line-buffered -v hostcfg; then echo "=== [e85] OK $(date) ==="; else echo "=== [e85] FAILED (exit $?) $(date) ==="; exit 1; fi
