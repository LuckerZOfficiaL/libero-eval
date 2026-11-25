#!/bin/bash
# 10-25 GPU $1: after its EXP-138 gate run (tmux e138g$1) and order 0's gate are done -> EXP-139 stage 1 (order 0) shard $1/3
# -> report. Usage: bash sweep/chain8_e139.sh 0|1|2
cd /home/lzhou/libero-eval; set -u; g=$1
case $g in
  0) export CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 ORCH_CUDA_MAP="0=GPU-8702ed42-9555-9082-d07a-3542fced5400" ORCH_EGL_MAP="0=0" E113_GPUS=0 E112_PORT0=8100 ;;
  1) export CUDA_VISIBLE_DEVICES=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e ORCH_CUDA_MAP="1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e" ORCH_EGL_MAP="1=1" E113_GPUS=1 E112_PORT0=8200 ;;
  2) export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba ORCH_CUDA_MAP="2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba" ORCH_EGL_MAP="2=0" E113_GPUS=2 ;;
esac
export E113_DEV=cuda:0
while tmux has-session -t e138g$g 2>/dev/null || ! grep -q "run DONE" results/exp138_s0.log; do sleep 120; done
echo "[g$g] e139 start $(date '+%m-%d %H:%M')"
for try in 1 2; do E139_SEQ=0 E139_SHARD=$g/3 E113_ROUND=1 .venv/bin/python -u sweep/exp139_embodiment.py && break; done
E139_SEQ=0 E139_STAGE=report .venv/bin/python -u sweep/exp139_embodiment.py 2>&1 | grep -v hostcfg > results/exp139_s0_report_console.txt
echo "[g$g] E139 EXIT $(date '+%m-%d %H:%M')"
