#!/bin/bash
# 11-06 EXP-147 cons1 (seed 1, order 0; trained on odin, new_loras_cons is on the shared NAS): eval once each GPU has <= 2 eval workers; port 9300.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis/new_loras_cons
echo "[c36] start $(date +%m-%d\ %H:%M)"
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
until grep -q "^\[done\]" $A/train_cons1.log && [ -f $A/train_cons1/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; do sleep 120; done
export E145_SEQ=0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=9300
E147_VARS=cons1 E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
for try in 1 2 3; do E147_VARS=cons1 E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
E147_VARS=cons,cons1 E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c36] cons1 EVAL EXIT $(date +%m-%d\ %H:%M)"
