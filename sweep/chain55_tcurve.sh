#!/bin/bash
# 11-08 EXP-152 t curve: the main pipeline (consfz, 3,000 steps) when only t = 2 / 4 new skills arrive (pseudo-orders 7-10 /
# 11-14 = orders 0-3 with their first t new skills, nested; exp116 TCUT). Lowest priority: each A6000 takes the next
# training once chain49 has claimed both of its trainings (after EXP-150's f5 / j5); claims .c55_f$s.
# Evals: consfz port 9650 (<= 2 eval workers per GPU before it starts); the training-free scratch merges (scr) port 9700,
# only in gaps (no other eval running), so they never stack 3 workers next to a trainer. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis; SEQS="7 8 9 10 11 12 13 14"
echo "[c55] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU
  until [ -f $A/.c49_r2 ] && [ -f $A/.c49_r3 ]; do sleep 120; done
  for s in $SEQS; do
    [ -f $A/.c55_f$s ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    ( set -o noclobber; : > $A/.c55_f$s ) 2>/dev/null || continue
    echo "[c55] trainer f$s on $1 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E147_VARS=consfz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  done
}
lane $U2 &
sleep 30; lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
left="$SEQS"; refs="$SEQS"
while [ -n "$left" ]; do
  for s in $left; do
    [ -f $A/new_loras_cons_s$s/train_consfz/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] || continue
    sleep 60   # adapter fully written
    for try in 1 2 3; do E112_PORT0=9650 E145_SEQ=$s E147_VARS=consfz E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    echo "[c55] CONSFZ s$s EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  done
  s=${refs%% *}
  if [ -n "$refs" ] && [ $(nev $U0) -eq 0 ] && [ $(nev $U2) -eq 0 ] && mkdir $A/.scr_lock 2>/dev/null; then   # one gap ref at a time (c55 / c56)
    for try in 1 2 3; do E112_PORT0=9700 E145_SEQ=$s E147_VARS=scr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    echo "[c55] REFS s$s EVAL EXIT $(date +%m-%d\ %H:%M)"; rmdir $A/.scr_lock
    refs=$(echo $refs | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  fi
  sleep 120
done
while [ -n "$refs" ]; do
  s=${refs%% *}
  until [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] && mkdir $A/.scr_lock 2>/dev/null; do sleep 120; done
  for try in 1 2 3; do E112_PORT0=9700 E145_SEQ=$s E147_VARS=scr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
  echo "[c55] REFS s$s EVAL EXIT $(date +%m-%d\ %H:%M)"; rmdir $A/.scr_lock
  refs=$(echo $refs | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
done
wait; echo "[c55] ALL EXIT $(date +%m-%d\ %H:%M)"
