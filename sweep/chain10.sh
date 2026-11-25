#!/bin/bash
# 10-25 GPU $1 (relaunch after the zero-factor KeyError; full dry build of every arm x layer passed): EXP-140 shard $1/3 ->
# EXP-139 (order 0) shard $1/3 -> both reports. Usage: bash sweep/chain10.sh 0|1|2
cd /home/lzhou/libero-eval; set -u; g=$1
case $g in
  0) export CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 ORCH_CUDA_MAP="0=GPU-8702ed42-9555-9082-d07a-3542fced5400" ORCH_EGL_MAP="0=0" E113_GPUS=0 E112_PORT0=8100 ;;
  1) export CUDA_VISIBLE_DEVICES=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e ORCH_CUDA_MAP="1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e" ORCH_EGL_MAP="1=1" E113_GPUS=1 E112_PORT0=8200 ;;
  2) export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba ORCH_CUDA_MAP="2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba" ORCH_EGL_MAP="2=0" E113_GPUS=2 ;;
esac
export E113_DEV=cuda:0
echo "[g$g] e140 start $(date '+%m-%d %H:%M')"
for try in 1 2; do E140_SHARD=$g/3 E113_ROUND=1 .venv/bin/python -u sweep/exp140_softgate.py && break; done
echo "[g$g] E140 SHARD EXIT $(date '+%m-%d %H:%M')"
for try in 1 2; do E139_SEQ=0 E139_SHARD=$g/3 E113_ROUND=1 .venv/bin/python -u sweep/exp139_embodiment.py && break; done
echo "[g$g] E139 SHARD EXIT $(date '+%m-%d %H:%M')"
