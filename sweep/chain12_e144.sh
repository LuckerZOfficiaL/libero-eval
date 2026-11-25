#!/bin/bash
# 10-26 EXP-144 pilot (data-free conversion through own rollouts), GPU $1 = shard $1/3 of order 0's 8 externals:
# [wait for this GPU's running jobs] -> collect own rollouts (3 workers) -> train WP adapters on them; GPU 1 then waits for
# all 3 training shards and runs base + merge on all GPUs + report. Usage: bash sweep/chain12_e144.sh 0|1|2
cd /home/lzhou/libero-eval; set -u; g=$1
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
case $g in
  0) export CUDA_VISIBLE_DEVICES=$U0 ORCH_CUDA_MAP="0=$U0" ORCH_EGL_MAP="0=0" E113_GPUS=0 E112_PORT0=8500
     until grep -q "E141X EXIT" results/chain11.log 2>/dev/null && grep -q "E139 SHARD EXIT" results/chain10_g0.log; do sleep 60; done ;;
  1) export CUDA_VISIBLE_DEVICES=$U1 ORCH_CUDA_MAP="1=$U1" ORCH_EGL_MAP="1=1" E113_GPUS=1 E112_PORT0=8600
     until grep -q "E139 SHARD EXIT" results/chain10_g1.log; do sleep 60; done ;;   # 10-26 00:28: /dev/shm too full to overlap
  2) export CUDA_VISIBLE_DEVICES=$U2 ORCH_CUDA_MAP="2=$U2" ORCH_EGL_MAP="2=0" E113_GPUS=2 E112_PORT0=8700
     until grep -q "E142 EXIT" results/e142.log 2>/dev/null && grep -q "E139 SHARD EXIT" results/chain10_g2.log; do sleep 60; done ;;
esac
export E113_DEV=cuda:0 E112_PER_GPU=2                                # 2 per GPU: each served adapter is a 6.8 GB /dev/shm build
echo "[g$g] e144 collect start $(date '+%m-%d %H:%M')"
for try in 1 2; do E144_STAGE=collect E144_SHARD=$g/3 .venv/bin/python -u sweep/exp144_selfconvert.py && break; done
echo "[g$g] e144 train start $(date '+%m-%d %H:%M')"
E144_STAGE=train E144_SHARD=$g/3 .venv/bin/python -u sweep/exp144_selfconvert.py
echo "[g$g] E144 TRAIN EXIT $(date '+%m-%d %H:%M')"
[ "$g" = 1 ] || exit 0
until [ $(grep -l "E144 TRAIN EXIT" results/chain12_g*.log 2>/dev/null | wc -l) -eq 3 ]; do sleep 60; done
export ORCH_CUDA_MAP="0=$U0,1=$U1,2=$U2" ORCH_EGL_MAP="0=0,1=1,2=0" E113_GPUS=0,1,2 E112_PORT0=8000
for st in base run; do for try in 1 2; do E144_STAGE=$st .venv/bin/python -u sweep/exp144_selfconvert.py && break; done; done
E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[g1] E144 ALL EXIT $(date '+%m-%d %H:%M')"
