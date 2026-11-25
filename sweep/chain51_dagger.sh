#!/bin/bash
# 11-07 EXP-155 DAgger round 2 (sweep/exp155_dagger.py), order 2 first (pre-registered gate before orders 0, 1, 3). Relabeling
# runs on odin (chain50) -> NAS. Training takes the first A6000 that frees up once order 2 is relabeled, ahead of chain48's
# remaining trainings (polls every 30 s and claims after 60 s free; chain48/49 lanes wait 180 s). Eval port 9550, 1 worker
# per GPU, <= 2 eval workers per GPU before it starts. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis; D=/media/pinas/lzhou/libero-eval-archive/exp144_dumps
echo "[c51] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU
  until [ $(ls $D/e155_*_s2/.done 2>/dev/null | wc -l) -eq 8 ]; do sleep 120; done
  [ -f $A/.c51_d2 ] && return
  until ! busy $1 && sleep 60 && ! busy $1; do sleep 30; done
  ( set -o noclobber; : > $A/.c51_d2 ) 2>/dev/null || return
  echo "[c51] trainer d2 on $1 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$1 E145_SEQ=2 E155_STAGE=train .venv/bin/python -u sweep/exp155_dagger.py 2>&1 | grep -v hostcfg
}
lane $U2 &
lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E145_SEQ=2
until [ -f $A/new_loras_cons_s2/train_consfzd/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; do sleep 120; done
sleep 60   # adapter fully written
for try in 1 2 3; do E112_PORT0=9550 E147_VARS=consfzd E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
E147_VARS=cons,consfz,consfz2,consfzd E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c51] ORDER 2 EVAL EXIT $(date +%m-%d\ %H:%M)"
wait; echo "[c51] ALL EXIT $(date +%m-%d\ %H:%M)"
