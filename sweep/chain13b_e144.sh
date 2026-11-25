#!/bin/bash
# 10-26 08:10 EXP-144 on GPU 2 after shiva GPU 1 went off the bus: every EGL client hangs in the NVIDIA modeset driver, so the
# simulator renders with OSMesa (CPU; ORCH_GL). Collect the 3 missing rollout sets -> OSMesa-vs-EGL check on the original skills
# -> train from the shared claim queue -> after GPU 0's queue (chain13_g0) ends: base + merge on GPUs 0 and 2 -> report.
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
export CUDA_VISIBLE_DEVICES=$U2 ORCH_CUDA_MAP="2=$U2" E113_GPUS=2 E112_PORT0=8700 E113_DEV=cuda:0 E112_PER_GPU=3 ORCH_GL=osmesa
echo "[g2] e144 collect start $(date '+%m-%d %H:%M')"
for try in 1 2; do E144_STAGE=collect .venv/bin/python -u sweep/exp144_selfconvert.py; done   # 2nd pass retries failed jobs
touch results/analysis/new_loras_s16wp_self/.collect_done
E144_STAGE=calib .venv/bin/python -u sweep/exp144_selfconvert.py
echo "[g2] e144 train start $(date '+%m-%d %H:%M')"
E144_STAGE=train .venv/bin/python -u sweep/exp144_selfconvert.py
echo "[g2] E144 TRAIN EXIT $(date '+%m-%d %H:%M')"
until grep -q "E144 TRAIN EXIT" results/chain13_g0.log 2>/dev/null; do sleep 60; done
export ORCH_CUDA_MAP="0=$U0,2=$U2" E113_GPUS=0,2 E112_PORT0=8000 E112_PER_GPU=2
for st in base run; do for try in 1 2; do E144_STAGE=$st .venv/bin/python -u sweep/exp144_selfconvert.py; done; done
E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[g2] E144 ALL EXIT $(date '+%m-%d %H:%M')"
