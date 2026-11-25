#!/bin/bash
# 11-05 EXP-145 generality: in-context trainings for orders 1-2 on odin's healthy P6000 (CUDA only; odin's EGL is dead since
# its GPU1 fault). Shared claim queue on the NAS (results/analysis/new_loras_ctx_s{n} -> NAS on odin and shiva). ctx first.
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-653b4880-3c1a-c9b6-7bcf-6cb199869263
echo "[c24] start $(date +%m-%d\ %H:%M)"
for v in ctx ctxr; do for s in 1 2; do E145_SEQ=$s E145_VARS=$v E145_STAGE=train .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg; done; done
echo "[c24] ALL EXIT $(date +%m-%d\ %H:%M)"
