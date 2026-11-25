#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
echo "=== [gto] strong 71,72,74,75 @10 START $(date) ==="
GT_TARGETS=71,72,74,75 GT_N=10 CHUNK=1 TORCHDYNAMO_DISABLE=1 .venv/bin/python sweep/gt_sweep.py
echo "=== [gto] EXIT $? $(date) ==="
