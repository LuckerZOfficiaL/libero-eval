#!/bin/bash
# 10-22 GPU 1 back (stress test passed): takes GPU 0's former shares -> scratch 3,4/5 (every step) -> wp 3,4/5 (t=2,4,6,8)
# -> MergeVLA-style -> restore units.
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e ORCH_CUDA_MAP="1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e" ORCH_EGL_MAP="1=1" E113_GPUS=1 E113_DEV=cuda:0 E112_PORT0=8200 TAG=g1
echo "[g1] start $(date '+%m-%d %H:%M')"
E134_SETTING=scratch E134_SHARD=3,4/5 E113_ROUND=1 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g1] SCRATCH SHARD EXIT $(date '+%m-%d %H:%M')"
E134_SETTING=wp E134_SKIP=retain E134_TS=2,4,6,8 E134_SHARD=3,4/5 E113_ROUND=1 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g1] WP SHARD EXIT $(date '+%m-%d %H:%M')"
E113_ROUND=1 .venv/bin/python -u sweep/exp135_mergevla.py
echo "[g1] MVLA EXIT $(date '+%m-%d %H:%M')"
E113_ROUND=1 bash sweep/restore_units.sh
