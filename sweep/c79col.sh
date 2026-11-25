#!/bin/bash
# 11-22 EXP-171 the 8 externals never drawn by orders 0-3: their own adapters' rollouts (EXP-144 protocol: 50 episodes, states
# 10..49, success-filtered dumps), under the eval lock, port 10960; then prep / prep2 for the 5th arrivals 33 / 34 (30 new).
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
echo "[c79col] start $(date +%m-%d\ %H:%M)"
( flock 9; export ORCH_CUDA_MAP="0=$G_U0,2=$G_U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=2 CUDA_VISIBLE_DEVICES=$G_U0 E113_DEV=cuda:0
  for try in 1 2; do E112_PORT0=10960 E144_SEQS=34 E144_STAGE=collect .venv/bin/python -u sweep/exp144_selfconvert.py 9>&- 2>&1 | grep -v hostcfg 9>&-; done ) 9> $A/.evlock
echo "[c79col] collect EXIT $(date +%m-%d\ %H:%M)"
for s in 34 33; do for st in prep prep2; do
  CUDA_VISIBLE_DEVICES= E145_SEQ=$s E147_VARS=consfzc E147_STAGE=$st .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg | grep 'e147\|Error\|assert'
done; done
echo "[c79col] ALL EXIT $(date +%m-%d\ %H:%M)"
