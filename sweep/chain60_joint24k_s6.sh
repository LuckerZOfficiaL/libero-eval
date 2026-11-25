#!/bin/bash
# 11-08 EXP-150 pre-registered fairness check: joint s6 library 78.2 fell > 5pp below joint s4 (85.0) -> joint at 24,000 steps
# on the 82 skills (E136_SFX=24k; rank 64, as joint s6). First A6000 free after chain57 has claimed f6x and chain53 j5 (60 s
# rule, ahead of chain49 / 55 / 56); claims .c60_j6x. ~25 h. Eval port 9950. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c60] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU
  until [ -f $A/.c57_f6x ] && [ -f $A/.c48_j5 ]; do sleep 60; done
  [ -f $A/.c60_j6x ] && return
  until ! busy $1 && sleep 60 && ! busy $1; do sleep 30; done
  ( set -o noclobber; : > $A/.c60_j6x ) 2>/dev/null || return
  echo "[c60] trainer j6x on $1 $(date +%H:%M)"
  E136_SFX=24k E136_STEPS=24000 E136_ORDERS=6 E136_GPU=$1 E136_STAGE=train .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg
}
lane $U0 &
sleep 20; lane $U2 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
until [ -f $A/joint_loras/joint24k_s6/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; do sleep 120; done
sleep 60   # adapter fully written
for try in 1 2; do E112_PORT0=9950 E136_SFX=24k E136_ORDERS=6 E136_STAGE=eval .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg; done
echo "[c60] JOINT24K s6 EVAL EXIT $(date +%m-%d\ %H:%M)"
wait; echo "[c60] ALL EXIT $(date +%m-%d\ %H:%M)"
