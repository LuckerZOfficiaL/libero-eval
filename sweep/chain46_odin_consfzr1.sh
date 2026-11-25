#!/bin/bash
# 11-07 seed check of the conversion-free, demo-free correction on order 0 (consfzr = 78.8 / 65.0 vs consfz 86.9 / 77.5 and
# 84.4 / 81.4 at 38 skills: variance or real?): consfzr seed 1 on odin's P6000 (~19 h; eval on shiva later).
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX
export CUDA_VISIBLE_DEVICES=GPU-653b4880-3c1a-c9b6-7bcf-6cb199869263 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 24 GB card: first try OOMed at 23.5 GB
echo "[c46] start $(date +%m-%d\ %H:%M)"
E145_SEQ=0 E147_VARS=consfzr1 E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c46] ALL EXIT $(date +%m-%d\ %H:%M)"
