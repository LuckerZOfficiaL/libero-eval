#!/bin/bash
# 10-24 GPU 1: after its current chain (chain4_gpu1.sh) ends -> EXP-137 shards (user: queue after the current runs).
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e ORCH_CUDA_MAP="1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e" ORCH_EGL_MAP="1=1" E113_GPUS=1 E113_DEV=cuda:0 E112_PORT0=8200 TAG=g1
while pgrep -f "bash sweep/chain4_gpu1.sh" >/dev/null; do sleep 300; done
echo "[g1] start $(date '+%m-%d %H:%M')"
bash sweep/e137_units.sh
