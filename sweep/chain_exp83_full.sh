#!/bin/bash
# EXP-83 full run: 2000 steps of meta-training, then the held-out rollout evaluation.
set -e
set -o pipefail
cd /home/lzhou/libero-eval
echo "=== [e83] FULL TRAIN $(date) ==="
.venv/bin/python sweep/exp83_train.py --steps 2000 --log_every 50 --save_every 200
echo "=== [e83] TRAIN OK $(date) ==="
echo "=== [e83] HELD-OUT EVAL $(date) ==="
.venv/bin/python sweep/exp83_eval.py
echo "=== [e83] ALL DONE $(date) ==="
