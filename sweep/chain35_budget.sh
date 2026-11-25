#!/bin/bash
# 11-06 EXP-147 step budget (order 0): consq / consh = cons at 750 / 1,500 steps (how much correction is needed?), trained on
# GPU2 after consr; consz6 = consz at 6,000 steps (can a longer correction replace conversion?) on GPU0 after consfz.
# Eval waits until each GPU has <= 2 eval workers (<= 3 workers + 1 trainer per A6000); ports 9400 / 9500. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis/new_loras_cons
echo "[c35] start $(date +%m-%d\ %H:%M)"
busy() { for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
roomy() { [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; }
export E145_SEQ=0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1
lane() {   # $1 GPU, $2 wait-for adapter, $3 arms, $4 port
  until [ -f $A/$2/adapter_model.safetensors ] && ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
  echo "[c35] trainer $3 on $1 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$1 E147_VARS=$3 E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  until roomy; do sleep 120; done
  export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=$4
  E147_VARS=$3 E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  for try in 1 2 3; do E147_VARS=$3 E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
  E147_VARS=cons,consz,$3 E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  echo "[c35] $3 EVAL EXIT $(date +%m-%d\ %H:%M)"
}
lane $U2 train_consr consq,consh 9400 &
lane $U0 train_consfz consz6 9500 &
wait; echo "[c35] ALL EXIT $(date +%m-%d\ %H:%M)"
