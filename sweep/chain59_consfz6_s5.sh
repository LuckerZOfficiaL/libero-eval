#!/bin/bash
# 11-08 EXP-150 dose check of the per-skill budget hypothesis: consfz at 6,000 steps on the 32-skill library (s5; consfz6) =
# the same replay samples per library skill as order 0 at 3,000 steps (~750). consfz s5 (3,000) = 77.5 / 68.8, order-0 library
# 86.9 (k 24) -> 80.0 (k 40) -> 76.9 (k 82). After chain57 has claimed f6x and chain53 j5 (60 s rule, ahead of chain49 / 55 / 56); claims
# .c59_f5x. Eval port 9900. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c59] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU
  until [ -f $A/.c57_f6x ] && [ -f $A/.c48_j5 ]; do sleep 60; done   # j5 (12 h) first: the longest pole
  [ -f $A/.c59_f5x ] && return
  until ! busy $1 && sleep 60 && ! busy $1; do sleep 30; done
  ( set -o noclobber; : > $A/.c59_f5x ) 2>/dev/null || return
  echo "[c59] trainer f5x on $1 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$1 E145_SEQ=5 E147_VARS=consfz6 E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
}
lane $U0 &
sleep 20; lane $U2 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
until [ -f $A/new_loras_cons_s5/train_consfz6/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; do sleep 120; done
sleep 60   # adapter fully written
for try in 1 2 3; do E112_PORT0=9900 E145_SEQ=5 E147_VARS=consfz6 E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
echo "[c59] CONSFZ6 s5 EVAL EXIT $(date +%m-%d\ %H:%M)"
wait; echo "[c59] ALL EXIT $(date +%m-%d\ %H:%M)"
