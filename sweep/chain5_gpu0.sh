#!/bin/bash
# 10-24 GPU 0 restart after the /dev/shm overflow (10:18): wait for its orphaned eval -> restore units.
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 ORCH_CUDA_MAP="0=GPU-8702ed42-9555-9082-d07a-3542fced5400" ORCH_EGL_MAP="0=0" E113_GPUS=0 E113_DEV=cuda:0 E112_PORT0=8100 TAG=g0
while kill -0 1435058 2>/dev/null; do sleep 60; done
echo "[g0] start $(date '+%m-%d %H:%M')"
E113_ROUND=1 bash sweep/restore_units.sh
