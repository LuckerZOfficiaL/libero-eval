#!/bin/bash
# 11-05 EXP-145d: both FedAvg means at once (no training) — conv + a (ctx - conv) + b (ctxr - conv); primary a = b = 1/8.
# 1 worker per GPU (GPUs 0+2 also hold the EXP-147 trainings and chain21's EXP-144 merges). GPU1 idle.
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 E112_PORT0=8300 E144_SEQS=0
echo "[c23] start $(date +%m-%d\ %H:%M)"
M="0.125:0.125,0.25:0.125"
E145_VARS=ctx,ctxr E145_BLEND=ctx E145_LAMS=0.25 E145_MIX=$M E145_STAGE=interp .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
for try in 1 2; do
  E145_VARS=ctx,ctxr E145_LAMS= E145_MIX=$M E145_STAGE=run .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
  E145_VARS=ctx,ctxr E145_BLEND=ctx E145_LAMS=0.25 E145_STAGE=run .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
done
E145_VARS=ctxr,ctx,ctxr_l0.125,ctxr_l0.25,ctx_l0.125,ctx_l0.25 E145_LAMS= E145_MIX=$M E145_STAGE=report .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
echo "[c23] E145D ALL EXIT $(date +%m-%d\ %H:%M)"
