#!/bin/bash
# 10-24 GPU 0: after its current chain (chain5_gpu0.sh) ends -> EXP-137 shards (user: queue after the current runs).
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 ORCH_CUDA_MAP="0=GPU-8702ed42-9555-9082-d07a-3542fced5400" ORCH_EGL_MAP="0=0" E113_GPUS=0 E113_DEV=cuda:0 E112_PORT0=8100 TAG=g0
while pgrep -f "bash sweep/chain5_gpu0.sh" >/dev/null; do sleep 300; done
echo "[g0] start $(date '+%m-%d %H:%M')"
bash sweep/e137_units.sh
