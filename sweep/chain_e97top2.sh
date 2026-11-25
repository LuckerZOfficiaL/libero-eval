#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
LA_ARMS=top2 LA_SLICE=0:7 CHUNK=6 .venv/bin/python sweep/la_operator_arms.py
echo "=== [e97top2] EXIT $? $(date) ==="
