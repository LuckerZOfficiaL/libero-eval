#!/bin/bash
# 11-08 EXP-151 continual arrival: the 22 new skills of the scaling order arrive per order (8 / 14 / 18 / 22, order 0's
# library); after each arrival the merge is rebuilt and the previous correction resumed for 1,500 steps (arm consfzc;
# E145_SEQ 19 / 20 / 4; arrival 1 = order 0's consfz). Baselines per arrival: one-shot consfz from scratch (f19 / f20; 8 and
# 22 exist) and the training-free merge (scr 19 / 20). After chain49's two trainings, ahead of chain55 (polls every 60 s and
# claims after 60 s free; chain55 waits 180 s); claims .c56_<job>. Evals port 9750 (<= 2 eval workers per GPU before it
# starts); scr refs port 9800, in gaps only (no other eval running). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis; JOBS="c19 c20 c4 f19 f20"
echo "[c56] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
arm() { [ ${1:0:1} = c ] && echo consfzc || echo consfz; }
out() { echo $A/new_loras_cons_s${1#?}/train_$(arm $1)/adapter_model.safetensors; }
ok() {   # job $1 can start: unclaimed, and a continual round only once the previous round's correction exists
  [ -f $A/.c56_$1 ] && return 1
  case $1 in c19) [ -f $A/new_loras_cons/train_consfz/adapter_model.safetensors ];; c20) [ -f $(out c19) ];; c4) [ -f $(out c20) ];; *) true;; esac
}
lane() {   # $1 GPU
  until [ -f $A/.c49_r2 ] && [ -f $A/.c49_r3 ]; do sleep 120; done
  while true; do
    left=0; ran=0
    for j in $JOBS; do
      [ -f $A/.c56_$j ] && continue; left=1
      ok $j || continue
      until ! busy $1 && sleep 60 && ! busy $1; do sleep 30; done
      ( set -o noclobber; : > $A/.c56_$j ) 2>/dev/null || continue
      echo "[c56] trainer $j on $1 $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$1 E145_SEQ=${j#?} E147_VARS=$(arm $j) E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
      ran=1; break
    done
    [ $left = 0 ] && break
    [ $ran = 0 ] && sleep 60
  done
}
lane $U2 &
sleep 30; lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
left="$JOBS"; refs="19 20"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] || continue
    sleep 60   # adapter fully written
    for try in 1 2 3; do E112_PORT0=9750 E145_SEQ=${j#?} E147_VARS=$(arm $j) E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    echo "[c56] $j EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//')
  done
  s=${refs%% *}
  if [ -n "$refs" ] && [ $(nev $U0) -eq 0 ] && [ $(nev $U2) -eq 0 ] && mkdir $A/.scr_lock 2>/dev/null; then   # one gap ref at a time (c55 / c56)
    for try in 1 2 3; do E112_PORT0=9800 E145_SEQ=$s E147_VARS=scr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    echo "[c56] REFS s$s EVAL EXIT $(date +%m-%d\ %H:%M)"; rmdir $A/.scr_lock
    refs=$(echo $refs | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  fi
  sleep 120
done
while [ -n "$refs" ]; do
  s=${refs%% *}
  until [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] && mkdir $A/.scr_lock 2>/dev/null; do sleep 120; done
  for try in 1 2 3; do E112_PORT0=9800 E145_SEQ=$s E147_VARS=scr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
  echo "[c56] REFS s$s EVAL EXIT $(date +%m-%d\ %H:%M)"; rmdir $A/.scr_lock
  refs=$(echo $refs | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
done
wait; echo "[c56] ALL EXIT $(date +%m-%d\ %H:%M)"
