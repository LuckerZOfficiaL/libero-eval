#!/bin/bash
# 11-18 meeting follow-up: trained baselines on OUR data (EXP-158), continual-learning baselines in the EXP-151 arrival stream
# (EXP-159) and the merge under the correction (EXP-160). Jobs = arm@order, claimed in priority order by two lanes (first free
# A6000, 60 s rule; claims .c67_<arm>_<order>); a continual round starts once the previous round's correction exists.
#   erl     ER from the library model (library-only merge + LoRA on new rollouts + library replay; no new adapter merged)
#   erlc    its continual rounds (orders 19 / 20 / 4 = 14 / 18 / 22 new; 1,500 steps, resumed)
#   consb   multi-task LoRA from the base on the same data (orders 1-3; order 0 exists)
#   seq     sequential fine-tuning from the library model on the arrival's new rollouts only; seqc = its rounds
#   consfa  consfz on the vanilla average; consfac = its rounds
#   consb64 consb at joint's recipe (rank 64, 12,000 steps), order 0
# Evals one arm at a time (2 workers per GPU), port 10350; training-free vanilla-average refs (sca) port 10400, in gaps. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
JOBS="erl@0 consb@1 erlc@19 erl@1 seq@0 consb@2 erlc@20 erl@2 seqc@19 consb@3 erlc@4 erl@3 seqc@20 consfa@0 seqc@4 consfac@19 consfac@20 consfac@4 consfa@1 consfa@2 consfa@3 consb64@0"
echo "[c67] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
dir() { [ $1 = 0 ] && echo $A/new_loras_cons || echo $A/new_loras_cons_s$1; }
out() { echo $(dir ${1#*@})/train_${1%@*}/adapter_model.safetensors; }
prev() {   # previous round of a continual job (empty for a first round)
  case $1 in *c@19) echo ${1%c@19}@0;; *c@20) echo ${1%@20}@19;; *c@4) echo ${1%@4}@20;; esac
}
ok() { [ -f $A/.c67_${1/@/_} ] && return 1; p=$(prev $1); [ -z "$p" ] || [ -f $(out $p) ]; }
lane() {   # $1 GPU
  while true; do
    left=0; ran=0
    for j in $JOBS; do
      [ -f $A/.c67_${j/@/_} ] && continue; left=1
      ok $j || continue
      until ! busy $1 && sleep 60 && ! busy $1; do sleep 30; done
      ( set -o noclobber; : > $A/.c67_${j/@/_} ) 2>/dev/null || continue
      echo "[c67] trainer $j on $1 $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$1 E145_SEQ=${j#*@} E147_VARS=${j%@*} E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
      ran=1; break
    done
    [ $left = 0 ] && break
    [ $ran = 0 ] && sleep 60
  done
}
lane $U0 &
sleep 30; lane $U2 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=2 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
left="$JOBS"; refs="0 19 20 4"
while [ -n "$left$refs" ]; do
  did=0
  for j in $left; do
    [ -f $(out $j) ] && [ $(nev $U0) -le 1 ] && [ $(nev $U2) -le 1 ] || continue
    sleep 60   # adapter fully written
    for try in 1 2 3; do E112_PORT0=10350 E145_SEQ=${j#*@} E147_VARS=${j%@*} E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    echo "[c67] $j EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); did=1; break
  done
  if [ $did = 0 ] && [ -n "$refs" ] && [ $(nev $U0) -eq 0 ] && [ $(nev $U2) -eq 0 ]; then
    s=${refs%% *}
    for try in 1 2 3; do E112_PORT0=10400 E145_SEQ=$s E147_VARS=sca E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    echo "[c67] sca@$s EVAL EXIT $(date +%m-%d\ %H:%M)"
    refs=$(echo $refs | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  fi
  [ $did = 0 ] && sleep 120
done
wait; echo "[c67] ALL EXIT $(date +%m-%d\ %H:%M)"
