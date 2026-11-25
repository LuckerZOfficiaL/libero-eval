#!/bin/bash
# EXP-92 replication: std vs top4 on fresh rollouts (records e97r_*), after the EXP-93 remainder
set -o pipefail
cd /home/lzhou/libero-eval
while tmux has-session -t e96r 2>/dev/null || pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
LA_PREFIX=e97r LA_ARMS=std,top4 CHUNK=6 .venv/bin/python sweep/la_operator_arms.py
echo "=== [e97rep] EXIT $? $(date) ==="
