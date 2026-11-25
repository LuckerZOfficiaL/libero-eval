#!/bin/bash
# 11-05 shiva rebooted after the 11-04 GPU1 fault. GPU1 left idle (fell off the bus twice under load: 10-26, 11-04).
# GPU2: last EXP-144 training (xb0e1f6) || GPUs 0+2: EXP-145c (no-replay FedAvg blend, ctx_l0.125) -> EXP-144 base + merges orders 1-3.
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
export E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2"
echo "[c21] start $(date +%m-%d\ %H:%M)"
( CUDA_VISIBLE_DEVICES=$U2 E144_SEQS=0,1,2,3 E144_STAGE=train .venv/bin/python -u sweep/exp144_selfconvert.py
  echo "[c21] E144 TRAIN EXIT $(date +%m-%d\ %H:%M)" ) &
( export CUDA_VISIBLE_DEVICES=$U0 E113_GPUS=0,2 E112_PER_GPU=2 E112_PORT0=8100 E144_SEQS=0 E145_BLEND=ctx E145_LAMS=0.125
  E145_VARS=ctx E145_STAGE=interp .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
  for try in 1 2; do E145_VARS=ctx_l0.125 E145_STAGE=run .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg; done
  E145_VARS=ctxr,ctx,ctx_l0.125 E145_LAMS=0.125,0.25 E145_BLEND=ctxr E145_STAGE=report .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
  echo "[c21] E145C ALL EXIT $(date +%m-%d\ %H:%M)" )
wait
export CUDA_VISIBLE_DEVICES=$U0 E112_PER_GPU=3 E113_GPUS=0,2 E112_PORT0=8000
for st in base run; do for try in 1 2; do E144_SEQS=1,2,3 E144_STAGE=$st .venv/bin/python -u sweep/exp144_selfconvert.py; done; done
E144_SEQS=0,1,2,3 E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[c21] ALL EXIT $(date +%m-%d\ %H:%M)"
