#!/bin/bash
# 10-24 GPU 2: after its current chain (chain5_gpu2.sh) ends -> EXP-137 shards (user: queue after the current runs).
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba ORCH_CUDA_MAP="2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba" ORCH_EGL_MAP="2=0" E113_GPUS=2 E113_DEV=cuda:0  TAG=g2
while pgrep -f "bash sweep/chain5_gpu2.sh" >/dev/null; do sleep 300; done
echo "[g2] start $(date '+%m-%d %H:%M')"
bash sweep/e137_units.sh
