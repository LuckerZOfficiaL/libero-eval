#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
LA_ARMS=b75,b125,b150 LA_SLICE=0:7 CHUNK=6 .venv/bin/python sweep/la_operator_arms.py
echo "=== [e96rest] EXIT $? $(date) ==="
