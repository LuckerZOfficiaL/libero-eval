#!/bin/bash
# 10-27 04:50 EXP-145 follow-up: FedAvg-style blends converted + lam (ctxr - converted), lam 1/8, 1/4 — merges only (evals),
# GPUs 0 and 2 (GPU 1 finishes chain15's last EXP-144 training), ports 8100+
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
export E113_DEV=cuda:0 E112_PER_GPU=2 CUDA_VISIBLE_DEVICES=$U0 ORCH_CUDA_MAP="0=$U0,1=$U1,2=$U2" ORCH_EGL_MAP="0=0,1=1,2=2" \
       E113_GPUS=0,2 E112_PORT0=8100 E144_SEQS=0
E145_STAGE=interp .venv/bin/python -u sweep/exp145_incontext.py
for try in 1 2; do E145_STAGE=run .venv/bin/python -u sweep/exp145_incontext.py; done
E145_STAGE=report .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
echo "[c17] E145B ALL EXIT $(date '+%m-%d %H:%M')"
