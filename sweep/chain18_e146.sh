#!/bin/bash
# 11-04 EXP-146 CUSD baseline: val (order 0, 5 settings) -> pick -> test (4 orders) -> report. All 3 GPUs, EGL. Queue: after
# EXP-144 (chain15 remainder) and EXP-145b (chain17).
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
export E113_DEV=cuda:0 E112_PER_GPU=2 CUDA_VISIBLE_DEVICES=$U0 ORCH_CUDA_MAP="0=$U0,1=$U1,2=$U2" ORCH_EGL_MAP="0=0,1=1,2=2" \
       E113_GPUS=0,1,2 E112_PORT0=8000
echo "[c18] val start $(date '+%m-%d %H:%M')"
for try in 1 2; do E146_STAGE=val .venv/bin/python -u sweep/exp146_cusd.py; done
E146_STAGE=pick .venv/bin/python -u sweep/exp146_cusd.py || exit 1
echo "[c18] test start $(date '+%m-%d %H:%M')"
for try in 1 2; do E146_STAGE=run .venv/bin/python -u sweep/exp146_cusd.py; done
E146_STAGE=report .venv/bin/python -u sweep/exp146_cusd.py 2>&1 | grep -v hostcfg
echo "[c18] E146 ALL EXIT $(date '+%m-%d %H:%M')"
