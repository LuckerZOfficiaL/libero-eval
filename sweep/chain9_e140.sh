#!/bin/bash
# 10-25 GPU $1: after its EXP-139 chain (chain8_e139.sh $1) -> EXP-140 shard $1/3 -> report. Usage: bash sweep/chain9_e140.sh 0|1|2
cd /home/lzhou/libero-eval; set -u; g=$1
case $g in
  0) export CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 ORCH_CUDA_MAP="0=GPU-8702ed42-9555-9082-d07a-3542fced5400" ORCH_EGL_MAP="0=0" E113_GPUS=0 E112_PORT0=8100 ;;
  1) export CUDA_VISIBLE_DEVICES=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e ORCH_CUDA_MAP="1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e" ORCH_EGL_MAP="1=1" E113_GPUS=1 E112_PORT0=8200 ;;
  2) export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba ORCH_CUDA_MAP="2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba" ORCH_EGL_MAP="2=0" E113_GPUS=2 ;;
esac
export E113_DEV=cuda:0
while pgrep -f "bash sweep/chain8_e139.sh $g" >/dev/null; do sleep 120; done
echo "[g$g] e140 start $(date '+%m-%d %H:%M')"
for try in 1 2; do E140_SHARD=$g/3 E113_ROUND=1 .venv/bin/python -u sweep/exp140_softgate.py && break; done
E140_STAGE=report .venv/bin/python -u sweep/exp140_softgate.py 2>&1 | grep -v hostcfg > results/exp140_report_console.txt
echo "[g$g] E140 EXIT $(date '+%m-%d %H:%M')"
