#!/bin/bash
# 11-06 EXP-147 seed check on the weakest order (order 2: cons new 73.8 vs joint 85.0): cons seed 1 on odin's P6000 (~18 h;
# eval on shiva later). Order 0 seed 1 landed within noise (86.2 / 73.8 vs 86.9 / 71.3).
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX
export CUDA_VISIBLE_DEVICES=GPU-653b4880-3c1a-c9b6-7bcf-6cb199869263
echo "[c39] start $(date +%m-%d\ %H:%M)"
E145_SEQ=2 E147_VARS=cons1 E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c39] ALL EXIT $(date +%m-%d\ %H:%M)"
