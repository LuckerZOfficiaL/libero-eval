#!/bin/bash
# 10-21 trimmed plan, 2 working GPUs (GPU 1 failed). GPU 0: joint training x4 -> joint eval -> 40% of the per-step runs.
# Everything pinned by UUID; simulator EGL rendering on device 0 (first in PCI order).
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 ORCH_CUDA_MAP="0=GPU-8702ed42-9555-9082-d07a-3542fced5400" ORCH_EGL_MAP="0=0" E113_GPUS=0 E113_DEV=cuda:0 E112_PORT0=8100
echo "[g0] start $(date '+%m-%d %H:%M')"
E136_STAGE=train E136_GPU=GPU-8702ed42-9555-9082-d07a-3542fced5400 .venv/bin/python -u sweep/exp136_joint.py
echo "[g0] JOINT TRAIN EXIT $(date '+%m-%d %H:%M')"
E136_STAGE=eval .venv/bin/python -u sweep/exp136_joint.py
echo "[g0] JOINT EVAL EXIT $(date '+%m-%d %H:%M')"
E134_SETTING=wp E134_SKIP=retain E134_SHARD=3,4/5 E113_ROUND=1 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g0] WP SHARD EXIT $(date '+%m-%d %H:%M')"
E134_SETTING=scratch E134_TS=2,4,6,8 E134_SHARD=3,4/5 E113_ROUND=1 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g0] SCRATCH SHARD EXIT $(date '+%m-%d %H:%M')"
