#!/bin/bash
# 11-06 EXP-147 consr eval rerun: chain33's worker 0 used port 9100 = shiva's Prometheus node exporter (server could not bind;
# clients got HTTP 200 -> parse_fail in 2 s). parse_fail records count as missing, so run fills only the gaps; port 9600.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c37] start $(date +%m-%d\ %H:%M)"
export E145_SEQ=0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=9600
for try in 1 2 3; do E147_VARS=consr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
E147_VARS=cons,consr E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c37] consr EVAL EXIT $(date +%m-%d\ %H:%M)"
