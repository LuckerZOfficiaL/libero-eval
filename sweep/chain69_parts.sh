#!/bin/bash
# 11-18 EXP-161 VLA part ablation, order 0, eval only: consfz's correction kept in one part (pv / pl / pe = vision / language /
# action expert + heads) or dropped from one (nv / nl / ne), merge unchanged. Gap evals, 1 worker per GPU, port 10500: only while
# <= 1 eval process per GPU and chain68's evals (port 10450) are not running -> at most 3 eval workers + 1 trainer per A6000.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c69] start $(date +%m-%d\ %H:%M)"
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
for v in pe pl pv ne nl nv; do
  until [ $(nev $U0) -le 1 ] && [ $(nev $U2) -le 1 ] && ! ss -ltn | grep -q ":1045[01] "; do sleep 120; done
  for try in 1 2 3; do E112_PORT0=10500 E145_SEQ=0 E147_VARS=$v E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
  echo "[c69] $v@0 EVAL EXIT $(date +%m-%d\ %H:%M)"
done
echo "[c69] ALL EXIT $(date +%m-%d\ %H:%M)"
