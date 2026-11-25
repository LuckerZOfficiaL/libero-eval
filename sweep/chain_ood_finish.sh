#!/bin/bash
# EXP-81 final cells on shiva. It died at 142/160 when EXP-83's oversized exports exhausted the quota;
# that is fixed. CHUNK=3 so it coexists with the EXP-83 evaluation on the same filesystem.
set -e
set -o pipefail
cd /home/lzhou/libero-eval
export CHUNK=3
echo "=== [ood] RESUME EXP-81 $(date) ==="
if .venv/bin/python sweep/ood_robust.py; then echo "=== [ood] OK EXP-81 $(date) ==="
else echo "=== [ood] FAILED EXP-81 (exit $?) $(date) ==="; exit 1; fi
