#!/bin/bash
# 11-05 EXP-147 (consolidation, order 0): GPU0 trains c0 + cons now; GPU2 trains consl once chain21's last EXP-144 training
# is done; then blend + merge evals on GPUs 0+2 (ports 8200+, alongside chain21's EXP-144 merges). GPU1 left idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c22] start $(date +%m-%d\ %H:%M)"
( CUDA_VISIBLE_DEVICES=$U0 E147_VARS=cons E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg ) &
( until grep -q "E144 TRAIN EXIT" results/chain21.log; do sleep 120; done
  CUDA_VISIBLE_DEVICES=$U2 E147_VARS=consl E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg ) &
wait
echo "[c22] E147 TRAIN EXIT $(date +%m-%d\ %H:%M)"
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=2 E112_PORT0=8200
E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
for try in 1 2; do E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c22] E147 ALL EXIT $(date +%m-%d\ %H:%M)"
