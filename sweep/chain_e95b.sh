#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
LS_SLICE=0:7 CHUNK=6 .venv/bin/python sweep/local_shared_swap.py
echo "=== [e95b] EXIT $? $(date) ==="
