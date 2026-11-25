#!/bin/bash
# 11-08 EXP-150 pre-registered fairness check, started early: consfz at 6,000 steps on the 74-skill library (s6; consfz6).
# consfz s6 (3,000 steps) = 78.1 / 68.8, order-0 library -10 vs k 24; joint s6 lands ~14:30 and if consfz breaks the 6,000-step
# run is required before concluding -> queued now: the first A6000 free after chain53 has claimed f5 (60 s rule, ahead of
# chain53's j5 / chain49 / chain55 / chain56 at 180 s or later); claims .c57_f6x. Eval port 9850. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c57] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU
  until [ -f $A/.c48_f5 ]; do sleep 60; done
  [ -f $A/.c57_f6x ] && return
  until ! busy $1 && sleep 60 && ! busy $1; do sleep 30; done
  ( set -o noclobber; : > $A/.c57_f6x ) 2>/dev/null || return
  echo "[c57] trainer f6x on $1 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$1 E145_SEQ=6 E147_VARS=consfz6 E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
}
lane $U0 &
sleep 20; lane $U2 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
until [ -f $A/new_loras_cons_s6/train_consfz6/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; do sleep 120; done
sleep 60   # adapter fully written
for try in 1 2 3; do E112_PORT0=9850 E145_SEQ=6 E147_VARS=consfz6 E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
echo "[c57] CONSFZ6 s6 EVAL EXIT $(date +%m-%d\ %H:%M)"
wait; echo "[c57] ALL EXIT $(date +%m-%d\ %H:%M)"
