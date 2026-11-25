#!/bin/bash
# 11-05 EXP-147: evaluate cons on GPU0 as soon as its training is done (consl still trains on GPU2 under chain22, whose
# later run stage then finds cons already recorded). GPU1 left idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400
echo "[c25] start $(date +%m-%d\ %H:%M)"
until [ -f results/analysis/new_loras_cons/train_cons/adapter_model.safetensors ]; do sleep 60; done; sleep 30
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0" ORCH_EGL_MAP="0=0" E113_GPUS=0 E112_PER_GPU=3 E112_PORT0=8400 E147_VARS=cons
E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
for try in 1 2; do E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c25] E147 CONS EXIT $(date +%m-%d\ %H:%M)"
