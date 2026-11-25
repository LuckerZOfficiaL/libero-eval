#!/bin/bash
# 10-25 GPU 1: replaces chain6_gpu1.sh (was only waiting). After its current chain (chain4_gpu1.sh) ends -> rerun the 4
# dropped add-on evaluations (parse_fail, transient; their units are done so nothing else retries them): ours w=0.25 order 1
# at t=6,7,8 and RETAIN-style order 3 at t=6 (items 25,26,27,137 of the wp t=1..8 list, 204 items) -> EXP-137 shards.
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e ORCH_CUDA_MAP="1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e" ORCH_EGL_MAP="1=1" E113_GPUS=1 E113_DEV=cuda:0 E112_PORT0=8200 TAG=g1
while pgrep -f "bash sweep/chain4_gpu1.sh" >/dev/null; do sleep 300; done
echo "[g1] fix start $(date '+%m-%d %H:%M')"
for try in 1 2; do
  E134_SETTING=wp E134_TS=1,2,3,4,5,6,7,8 E134_SHARD=25,26,27,137/204 E113_ROUND=1 .venv/bin/python -u sweep/exp134_stepwise.py && break
done
echo "[g1] FIX EXIT $(date '+%m-%d %H:%M')"
bash sweep/e137_units.sh
