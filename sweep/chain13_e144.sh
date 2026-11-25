#!/bin/bash
# 10-26 EXP-144 pilot resumed on GPUs 0 and 2 (shiva GPU 1 fell off the bus ~05:03 and hung chain12's base stage; chain12's
# shards had also skipped 3 collections): GPU 2 collects the missing rollout sets while GPU 0 starts training; both then train
# from a shared claim queue; GPU 0 then runs base + merge on both GPUs + report. Usage: bash sweep/chain13_e144.sh 0|2
cd /home/lzhou/libero-eval; set -u; g=$1
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
case $g in
  0) export CUDA_VISIBLE_DEVICES=$U0 ORCH_CUDA_MAP="0=$U0" ORCH_EGL_MAP="0=0" E113_GPUS=0 E112_PORT0=8500 ;;
  2) export CUDA_VISIBLE_DEVICES=$U2 ORCH_CUDA_MAP="2=$U2" ORCH_EGL_MAP="2=0" E113_GPUS=2 E112_PORT0=8700 ;;
esac
export E113_DEV=cuda:0 E112_PER_GPU=2                                # 2 per GPU: each served adapter is a 6.8 GB /dev/shm build
if [ "$g" = 2 ]; then
  echo "[g2] e144 collect start $(date '+%m-%d %H:%M')"
  for try in 1 2; do E144_STAGE=collect .venv/bin/python -u sweep/exp144_selfconvert.py; done   # 2nd pass retries failed jobs
  touch results/analysis/new_loras_s16wp_self/.collect_done
fi
echo "[g$g] e144 train start $(date '+%m-%d %H:%M')"
E144_STAGE=train .venv/bin/python -u sweep/exp144_selfconvert.py
echo "[g$g] E144 TRAIN EXIT $(date '+%m-%d %H:%M')"
[ "$g" = 0 ] || exit 0
until grep -q "E144 TRAIN EXIT" results/chain13_g2.log 2>/dev/null; do sleep 60; done
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=0" E113_GPUS=0,2 E112_PORT0=8000
for st in base run; do for try in 1 2; do E144_STAGE=$st .venv/bin/python -u sweep/exp144_selfconvert.py; done; done
E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[g0] E144 ALL EXIT $(date '+%m-%d %H:%M')"
