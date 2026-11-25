#!/bin/bash
# odin: top2 for targets 0:7 (e97)
set -o pipefail
cd /home/lzhou/libero-eval
LA_ARMS=top2 LA_SLICE=0:7 CHUNK=1 TORCHDYNAMO_DISABLE=1 .venv/bin/python sweep/la_operator_arms.py
echo "=== [e97odin] EXIT $? $(date) ==="
