#!/bin/bash
# 11-06 EXP-148 scaling: order 0's 16 library skills + ALL 22 new skills (E145_SEQ=4, pseudo-order C.SCALE).
#  now   : training-free references (scr = scratch merge 1/k, convm = converted merge) on idle eval slots, port 9800
#  later : one correction (cons, 3,000 steps) + joint training on all 38 skills (rank 64, 12k steps), each on the first A6000
#          free after chain38 has claimed consr orders 1-3 (claim files split the two jobs); evals ports 9900 / 9950.
# <= 2 eval workers per GPU before any eval starts (<= 3 workers + 1 trainer per A6000). GPU1 idle. Never port 9100.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis; D=$A/new_loras_cons_s4
echo "[c40] start $(date +%m-%d\ %H:%M)"
busy() { for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
roomy() { [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; }
export E145_SEQ=4 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1
claimed() { [ -f $A/new_loras_cons_s1/.claim_consr ] && [ -f $A/new_loras_cons_s2/.claim_consr ] && [ -f $A/new_loras_cons_s3/.claim_consr ]; }
lane() {   # $1 GPU: take cons s4, else joint s4, whichever is unclaimed
  until claimed && ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
  if ( set -o noclobber; : > $D/.claim_lane_cons ) 2>/dev/null; then
    echo "[c40] trainer cons s4 on $1 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 E147_VARS=cons E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  elif ( set -o noclobber; : > $D/.claim_lane_joint ) 2>/dev/null; then
    echo "[c40] trainer joint s4 on $1 $(date +%H:%M)"
    E136_ORDERS=4 E136_GPU=$1 E136_STAGE=train .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg
  fi
}
lane $U2 &
lane $U0 &
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
until roomy; do sleep 120; done
for try in 1 2 3; do E112_PORT0=9800 E147_VARS=scr,convm E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
E147_VARS=scr,convm E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c40] REFS EVAL EXIT $(date +%m-%d\ %H:%M)"
cdone=0; jdone=0
while [ $cdone = 0 ] || [ $jdone = 0 ]; do
  if [ $cdone = 0 ] && [ -f $D/train_cons/adapter_model.safetensors ] && roomy; then
    sleep 60
    E112_PORT0=9900 E147_VARS=cons E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    for try in 1 2 3; do E112_PORT0=9900 E147_VARS=cons E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    E147_VARS=scr,convm,cons E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c40] CONS EVAL EXIT $(date +%m-%d\ %H:%M)"; cdone=1
  fi
  if [ $jdone = 0 ] && [ -f $A/joint_loras/joint_s4/adapter_model.safetensors ] && roomy; then
    sleep 60
    for try in 1 2; do E112_PORT0=9950 E136_ORDERS=4 E136_STAGE=eval .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg; done
    echo "[c40] JOINT EVAL EXIT $(date +%m-%d\ %H:%M)"; jdone=1
  fi
  sleep 120
done
wait; echo "[c40] ALL EXIT $(date +%m-%d\ %H:%M)"
