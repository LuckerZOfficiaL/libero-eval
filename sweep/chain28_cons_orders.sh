#!/bin/bash
# 11-05 EXP-147 generality: consolidation (cons, 3,000 steps) for orders 1-3 on shiva (replaces the 16-training FedAvg round:
# cons 90.8 / 89.4 on order 0). Trainers: GPU0 now, GPU2 once consl is trained (claims per order dir on the NAS). Evaluator:
# each order as soon as its correction exists, 1 worker per GPU on 0+2 (<= 3 workers + 1 trainer per A6000). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c28] start $(date +%m-%d\ %H:%M)"
trainer() { for s in 1 2 3; do CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E147_VARS=cons E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg | sed "s/^/[s$s] /"; done; }
( trainer $U0 ) &
( until [ -f results/analysis/new_loras_cons/train_consl/adapter_model.safetensors ]; do sleep 60; done; sleep 60; trainer $U2 ) &
( export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 E112_PORT0=8600 E147_VARS=cons
  for s in 1 2 3; do
    until [ -f results/analysis/new_loras_cons_s$s/train_cons/adapter_model.safetensors ]; do sleep 120; done; sleep 30
    E145_SEQ=$s E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    for try in 1 2 3; do E145_SEQ=$s E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    E145_SEQ=$s E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c28] ORDER $s EVAL EXIT $(date +%m-%d\ %H:%M)"
  done ) &
wait
echo "[c28] ALL EXIT $(date +%m-%d\ %H:%M)"
