#!/bin/bash
# 10-25 completeness reruns (status parse_fail in 3 s, transient; their units / shards had finished, so nothing retried them).
#   g0: add-on c134ws2t4-retain, c134ws3t7-opcm (items 74,127 of the wp t=1..8 list, 204 items)
#   g2: EXP-137 c137s3t4-dilute (2 scenes), c137s0t8-dilute (items 11, 20 of 24)
# Usage: bash sweep/fix8_reruns.sh g0|g2. Ends with "[$TAG] FIX8 EXIT".
cd /home/lzhou/libero-eval; set -u
if [ "$1" = g0 ]; then
  export CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 ORCH_CUDA_MAP="0=GPU-8702ed42-9555-9082-d07a-3542fced5400" ORCH_EGL_MAP="0=0" E113_GPUS=0 E113_DEV=cuda:0 E112_PORT0=8100 TAG=g0
  for try in 1 2; do E134_SETTING=wp E134_TS=1,2,3,4,5,6,7,8 E134_SHARD=74,127/204 E113_ROUND=1 .venv/bin/python -u sweep/exp134_stepwise.py && break; done
else
  export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba ORCH_CUDA_MAP="2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba" ORCH_EGL_MAP="2=0" E113_GPUS=2 E113_DEV=cuda:0 TAG=g2
  for k in 11 20; do for try in 1 2; do E137_SHARD=$k/24 E113_ROUND=1 .venv/bin/python -u sweep/exp137_dilution.py && break; done; done
fi
echo "[$TAG] FIX8 EXIT $(date '+%m-%d %H:%M')"
