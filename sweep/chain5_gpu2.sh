#!/bin/bash
# 10-24 GPU 2 restart after the /dev/shm overflow (10:12): wait for its orphaned eval -> rest of wp shard 0,1,2/5 -> restore units.
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba ORCH_CUDA_MAP="2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba" ORCH_EGL_MAP="2=0" E113_GPUS=2 E113_DEV=cuda:0 TAG=g2
while kill -0 1431641 2>/dev/null; do sleep 60; done
echo "[g2] start $(date '+%m-%d %H:%M')"
for try in 1 2 3; do
  E134_SETTING=wp E134_SKIP=retain E134_TS=2,4,6,8 E134_SHARD=0,1,2/5 E113_ROUND=2 .venv/bin/python -u sweep/exp134_stepwise.py && break
  echo "[g2] WP SHARD FAILED (try $try) $(date '+%m-%d %H:%M')"; sleep 600
done
echo "[g2] WP SHARD EXIT $(date '+%m-%d %H:%M')"
E113_ROUND=2 bash sweep/restore_units.sh
