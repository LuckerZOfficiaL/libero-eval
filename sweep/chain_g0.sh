#!/bin/bash
# 10-21 plan, shiva GPU 0: joint-training upper bound (4 trainings), its evaluation, then 1/3 of the per-step runs.
cd /home/lzhou/libero-eval; set -u
until grep -q "\[e132\] REPORT EXIT" results/exp132.log 2>/dev/null; do sleep 120; done
echo "[g0] start $(date '+%m-%d %H:%M')"
E136_STAGE=train E136_GPU=0 .venv/bin/python -u sweep/exp136_joint.py
echo "[g0] JOINT TRAIN EXIT $(date '+%m-%d %H:%M')"
E136_STAGE=eval E113_GPUS=0 E113_DEV=cuda:0 E112_PORT0=8100 .venv/bin/python -u sweep/exp136_joint.py
echo "[g0] JOINT EVAL EXIT $(date '+%m-%d %H:%M')"
E134_SETTING=wp E134_SHARD=0/4 E113_GPUS=0 E113_DEV=cuda:0 E112_PORT0=8100 E113_ROUND=1 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g0] WP SHARD EXIT $(date '+%m-%d %H:%M')"
E134_SETTING=scratch E134_SHARD=0/4 E113_GPUS=0 E113_DEV=cuda:0 E112_PORT0=8100 E113_ROUND=1 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g0] SCRATCH SHARD EXIT $(date '+%m-%d %H:%M')"
