#!/bin/bash
# 10-21 plan, shiva GPUs 1,2: 2/3 of the per-step runs (write-protected), MergeVLA-style, 2/3 of the per-step runs (scratch).
cd /home/lzhou/libero-eval; set -u
until grep -q "\[e132\] REPORT EXIT" results/exp132.log 2>/dev/null; do sleep 120; done
rm -rf /dev/shm/mergesweep/cache
echo "[g12] start $(date '+%m-%d %H:%M')"
E134_SETTING=wp E134_SHARD=1,2,3/4 E113_GPUS=1,2 E113_DEV=cuda:1 E113_ROUND=2 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g12] WP SHARD EXIT $(date '+%m-%d %H:%M')"
E113_GPUS=1,2 E113_DEV=cuda:2 E113_ROUND=2 .venv/bin/python -u sweep/exp135_mergevla.py
echo "[g12] MVLA EXIT $(date '+%m-%d %H:%M')"
E134_SETTING=scratch E134_SHARD=1,2,3/4 E113_GPUS=1,2 E113_DEV=cuda:1 E113_ROUND=2 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g12] SCRATCH SHARD EXIT $(date '+%m-%d %H:%M')"
