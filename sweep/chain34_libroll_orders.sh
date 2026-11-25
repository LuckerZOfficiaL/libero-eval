#!/bin/bash
# 11-06 EXP-147 consr prep for orders 1-3: collect the library skills' own rollouts (skills shared with order 0 are skipped by
# record) on idle eval slots (1 worker per GPU on 0+2, port 9200; <= 3 workers + 1 trainer per A6000). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c34] start $(date +%m-%d\ %H:%M)"
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 E112_PORT0=9200
for s in 1 2 3; do E145_SEQ=$s E147_VARS=consr E147_STAGE=collect_lib .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep "e147\]"; done
echo "[c34] ALL EXIT $(date +%m-%d\ %H:%M)"
