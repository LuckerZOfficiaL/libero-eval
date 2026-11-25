#!/bin/bash
# 11-07 EXP-150 bigger library: E145_SEQ 5 (order 0's 16 + 16 seeded LIBERO-90 skills = 32, + order 0's 8 new) and 6 (all 74
# fleet skills + the same 8 new); libraries nested. Same recipes as EXP-148 (consfz 3,000 steps; joint rank 64, 12k steps).
#  now   : training-free references (scr = scratch merge 1/k) on the free eval slots, port 9400, 1 worker per GPU
#  later : each A6000 takes the next of consfz s6, joint s6, consfz s5, joint s5 once chain47 has claimed all 4 round-2
#          trainings; evals port 9450 (consfz) / 9500 (joint). <= 2 eval workers per GPU before an eval starts. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c48] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
roomy() { [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; }
lane() {   # $1 GPU
  until [ -f $A/.c47_t2 ] && [ -f $A/.c47_t0 ] && [ -f $A/.c47_t1 ] && [ -f $A/.c47_t3 ]; do sleep 120; done
  for j in f6 j6 f5 j5; do
    [ -f $A/.c48_$j ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    ( set -o noclobber; : > $A/.c48_$j ) 2>/dev/null || continue
    s=${j#?}; echo "[c48] trainer $j on $1 $(date +%H:%M)"
    if [ ${j:0:1} = f ]; then CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E147_VARS=consfz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    else E136_ORDERS=$s E136_GPU=$1 E136_STAGE=train .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg; fi
  done
}
lane $U2 &
lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
until roomy; do sleep 120; done
for s in 5 6; do
  for try in 1 2 3; do E112_PORT0=9400 E145_SEQ=$s E147_VARS=scr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
  echo "[c48] REFS s$s EVAL EXIT $(date +%m-%d\ %H:%M)"
done
left="f6 j6 f5 j5"
while [ -n "$left" ]; do
  for j in $left; do
    s=${j#?}
    if [ ${j:0:1} = f ]; then f=$A/new_loras_cons_s$s/train_consfz/adapter_model.safetensors; else f=$A/joint_loras/joint_s$s/adapter_model.safetensors; fi
    [ -f $f ] && roomy || continue
    sleep 60   # adapter fully written
    if [ ${j:0:1} = f ]; then
      for try in 1 2 3; do E112_PORT0=9450 E145_SEQ=$s E147_VARS=consfz E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
      echo "[c48] CONSFZ s$s EVAL EXIT $(date +%m-%d\ %H:%M)"
    else
      for try in 1 2; do E112_PORT0=9500 E136_ORDERS=$s E136_STAGE=eval .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg; done
      echo "[c48] JOINT s$s EVAL EXIT $(date +%m-%d\ %H:%M)"
    fi
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//')
  done
  sleep 120
done
wait; echo "[c48] ALL EXIT $(date +%m-%d\ %H:%M)"
