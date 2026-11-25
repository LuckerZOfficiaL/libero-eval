#!/bin/bash
# 10-26 19:55 EXP-145 pilot (order 0): in-context conversion (ctxr, then ctx), 3 trainers sharing the GPUs with chain15's EXP-144
# trainers; then the two merges (orchestrator ports 8100+, 1 server per GPU next to chain15's evals) + report
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
export E113_DEV=cuda:0 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 ORCH_CUDA_MAP="0=$U0,1=$U1,2=$U2" ORCH_EGL_MAP="0=0,1=1,2=2" \
       E113_GPUS=0,1,2 E112_PORT0=8100 E144_SEQS=0
echo "[c16] train start $(date '+%m-%d %H:%M')"
for u in $U0 $U1 $U2; do CUDA_VISIBLE_DEVICES=$u E145_STAGE=train .venv/bin/python -u sweep/exp145_incontext.py & sleep 30; done; wait
echo "[c16] E145 TRAIN EXIT $(date '+%m-%d %H:%M')"
for try in 1 2; do E145_STAGE=run .venv/bin/python -u sweep/exp145_incontext.py; done
E145_STAGE=report .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
echo "[c16] E145 ALL EXIT $(date '+%m-%d %H:%M')"
