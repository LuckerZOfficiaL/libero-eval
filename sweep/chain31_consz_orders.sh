#!/bin/bash
# 11-06 EXP-147 consz generality: the correction on the FROM-SCRATCH merge (no per-skill conversion) for orders 1-3. Order 0:
# consz 90.2 / 75.7 vs cons 90.8 / 89.4 -> if it holds, the method = training-free merge + one 3,000-step correction. Trainers:
# each GPU once chain29 has no ablation left for it (all 4 claims taken, GPU idle twice 3 min apart). Evaluator: each order as
# soon as trained, 1 worker per GPU on 0+2 (port 8800; <= 3 workers + 1 trainer per A6000). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c31] start $(date +%m-%d\ %H:%M)"
busy() { for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
claimed() { for v in consz consf conso consfz; do [ -f $A/new_loras_cons/.claim_$v ] || return 1; done; }
for U in $U0 $U2; do
  ( until claimed && ! busy $U && sleep 180 && ! busy $U; do sleep 120; done; echo "[c31] trainer on $U $(date +%H:%M)"
    for s in 1 2 3; do CUDA_VISIBLE_DEVICES=$U E145_SEQ=$s E147_VARS=consz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg | sed "s/^/[s$s] /"; done ) &
done
( export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 E112_PORT0=8800
  for s in 1 2 3; do
    until [ -f $A/new_loras_cons_s$s/train_consz/adapter_model.safetensors ]; do sleep 120; done; sleep 30
    for try in 1 2 3; do E145_SEQ=$s E147_VARS=consz E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    E145_SEQ=$s E147_VARS=cons,consz E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c31] ORDER $s EVAL EXIT $(date +%m-%d\ %H:%M)"
  done ) &
wait
echo "[c31] ALL EXIT $(date +%m-%d\ %H:%M)"
