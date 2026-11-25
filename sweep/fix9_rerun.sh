#!/bin/bash
# 10-25 completeness rerun: EXP-140 c140s1t8-soft_w1 (item 7 of 12) lost 1 scene to a transient parse_fail; its shard had
# finished. Runs on GPU1 next to the EXP-139 shard (own ports 8300+, 1 worker). Ends with "[g1] FIX9 EXIT".
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e ORCH_CUDA_MAP="1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e" ORCH_EGL_MAP="1=1" E113_GPUS=1 E113_DEV=cuda:0 E112_PORT0=8300 E112_PER_GPU=1
for try in 1 2; do E140_SHARD=7/12 E113_ROUND=1 .venv/bin/python -u sweep/exp140_softgate.py && break; done
echo "[g1] FIX9 EXIT $(date '+%m-%d %H:%M')"
