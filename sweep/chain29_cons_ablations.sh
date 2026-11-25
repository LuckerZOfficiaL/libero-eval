#!/bin/bash
# 11-05 EXP-147 ablations (order 0), in priority order: consz (cons trained on the FROM-SCRATCH merge: is the per-skill
# conversion needed?), consf (plain-LoRA fine-tune of the converted merge, same data + steps: does write protection matter?),
# conso (own rollouts only, no replay), consfz (plain LoRA on the scratch merge). Each GPU picks them up once chain28 has no
# order left for it (s1, s2 trained, s3 claimed) and no trainer runs on it. Evals: 1 worker per GPU on 0+2 (port 8700). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c29] start $(date +%m-%d\ %H:%M)"
busy() { for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
q() { for v in consz consf conso consfz; do CUDA_VISIBLE_DEVICES=$1 E145_SEQ=0 E147_VARS=$v E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done; }
for U in $U0 $U2; do
  ( until [ -f $A/new_loras_cons_s1/train_cons/adapter_model.safetensors ] && [ -f $A/new_loras_cons_s2/train_cons/adapter_model.safetensors ] && [ -f $A/new_loras_cons_s3/.claim_cons ]; do sleep 120; done
    sleep 180; while busy $U; do sleep 120; done; echo "[c29] trainer on $U $(date +%H:%M)"; q $U ) &
done
( export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 E112_PORT0=8700 E145_SEQ=0
  for v in consz consf conso consfz; do
    until [ -f $A/new_loras_cons/train_$v/adapter_model.safetensors ]; do sleep 120; done; sleep 30
    E147_VARS=$v E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    for try in 1 2 3; do E147_VARS=$v E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    E147_VARS=cons,consl,conso,consz,consf,consfz E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c29] $v EVAL EXIT $(date +%m-%d\ %H:%M)"
  done ) &
wait
echo "[c29] ALL EXIT $(date +%m-%d\ %H:%M)"
