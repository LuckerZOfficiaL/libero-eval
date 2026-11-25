#!/bin/bash
# 11-10 EXP-150 capacity check, seed 1: consfz6 seed 0 / 1 = library 80.9 / 84.2 (3.3pp seed spread > the single-run
# criterion's margin) -> rank 64 at seed 1 (consfz6r64s1) so the capacity test reads 2 seeds vs 2 seeds. First A6000 free
# (60 s rule); claims .c65_f6q. ~6.5 h training + ~3 h eval. Eval port 10250. GPU1 idle.
#
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c65] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU
  [ -f $A/.c65_f6q ] && return
  until ! busy $1 && sleep 60 && ! busy $1; do [ -f $A/.c65_f6q ] && return; sleep 30; done
  ( set -o noclobber; : > $A/.c65_f6q ) 2>/dev/null || return
  echo "[c65] trainer f6q on $1 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$1 E145_SEQ=6 E147_VARS=consfz6r64s1 E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
}
lane $U0 &
sleep 20; lane $U2 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
until [ -f $A/new_loras_cons_s6/train_consfz6r64s1/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; do sleep 120; done
sleep 60   # adapter fully written
for try in 1 2 3; do E112_PORT0=10250 E145_SEQ=6 E147_VARS=consfz6r64s1 E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
echo "[c65] CONSFZ6R64S1 s6 EVAL EXIT $(date +%m-%d\ %H:%M)"
wait; echo "[c65] ALL EXIT $(date +%m-%d\ %H:%M)"
