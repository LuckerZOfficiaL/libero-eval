#!/bin/bash
# 11-05 EXP-147 seed check: cons (order 0) retrained with seed 1 on odin's healthy P6000 (CUDA only, ~18 h; evals on shiva).
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX
export CUDA_VISIBLE_DEVICES=GPU-653b4880-3c1a-c9b6-7bcf-6cb199869263
echo "[c30] start $(date +%m-%d\ %H:%M)"
E145_SEQ=0 E147_VARS=cons1 E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c30] ALL EXIT $(date +%m-%d\ %H:%M)"
