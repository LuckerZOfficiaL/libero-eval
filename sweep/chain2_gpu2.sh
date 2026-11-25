#!/bin/bash
# 10-21 trimmed plan, 2 working GPUs (GPU 1 failed). GPU 2: seed merges -> 60% of the per-step runs + MergeVLA-style.
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba ORCH_CUDA_MAP="2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba" ORCH_EGL_MAP="2=0" E113_GPUS=2 E113_DEV=cuda:0
rm -rf /dev/shm/mergesweep/cache /dev/shm/mergesweep/prebuilt/e1*
echo "[g2] start $(date '+%m-%d %H:%M')"
E132_STAGE=base E113_ROUND=2 .venv/bin/python -u sweep/exp132_seeds.py
E113_ROUND=2 .venv/bin/python -u sweep/exp132_seeds.py
E132_STAGE=report .venv/bin/python -u sweep/exp132_seeds.py 2>&1 | grep -v hostcfg | tee results/exp132_report_console.txt
echo "[e132] REPORT EXIT $(date '+%m-%d %H:%M')" >> results/exp132.log; echo "[g2] SEED EXIT $(date '+%m-%d %H:%M')"
E134_SETTING=wp E134_SKIP=retain E134_SHARD=0,1,2/5 E113_ROUND=2 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g2] WP SHARD EXIT $(date '+%m-%d %H:%M')"
E113_ROUND=2 .venv/bin/python -u sweep/exp135_mergevla.py
echo "[g2] MVLA EXIT $(date '+%m-%d %H:%M')"
E134_SETTING=scratch E134_TS=2,4,6,8 E134_SHARD=0,1,2/5 E113_ROUND=2 .venv/bin/python -u sweep/exp134_stepwise.py
echo "[g2] SCRATCH SHARD EXIT $(date '+%m-%d %H:%M')"
