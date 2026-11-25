#!/bin/bash
# 10-22 GPU 0: wait for the running joint training (4 orders) -> joint eval -> restore units.
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 ORCH_CUDA_MAP="0=GPU-8702ed42-9555-9082-d07a-3542fced5400" ORCH_EGL_MAP="0=0" E113_GPUS=0 E113_DEV=cuda:0 E112_PORT0=8100 TAG=g0
while pgrep -f "sweep/exp136_joint.py" >/dev/null; do sleep 300; done
E136_STAGE=train E136_GPU=GPU-8702ed42-9555-9082-d07a-3542fced5400 .venv/bin/python -u sweep/exp136_joint.py
echo "[g0] JOINT TRAIN EXIT $(date '+%m-%d %H:%M')"
E136_STAGE=eval .venv/bin/python -u sweep/exp136_joint.py
echo "[g0] JOINT EVAL EXIT $(date '+%m-%d %H:%M')"
E113_ROUND=1 bash sweep/restore_units.sh
