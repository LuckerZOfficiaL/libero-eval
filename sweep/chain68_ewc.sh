#!/bin/bash
# 11-18 EXP-159 online-EWC baseline in the arrival stream: ewc@0 (sequential fine-tuning from the library model + EWC, Fisher of
# the library replay at the start; lambda 1e-3 pre-registered) -> ewcc@19 / 20 / 4 (resumed, previous ewc.pt). Starts once
# chain67 has claimed seq@0 (EWC sits with the other CL baselines), then competes for the first free A6000 (60 s rule); claims
# .c68_<arm>_<order>. Evals 1 worker per GPU, port 10450, only while <= 1 other eval process per GPU (chain67 uses 2 per GPU
# -> at most 3 eval workers + 1 trainer per A6000). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
JOBS="ewc@0 ewcc@19 ewcc@20 ewcc@4"
echo "[c68] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
dir() { [ $1 = 0 ] && echo $A/new_loras_cons || echo $A/new_loras_cons_s$1; }
out() { echo $(dir ${1#*@})/train_${1%@*}/adapter_model.safetensors; }
prev() { case $1 in *c@19) echo ${1%c@19}@0;; *c@20) echo ${1%@20}@19;; *c@4) echo ${1%@4}@20;; esac; }
ok() { [ -f $A/.c68_${1/@/_} ] && return 1; p=$(prev $1); [ -z "$p" ] || [ -f $(out $p) ]; }
until [ -f $A/.c67_seq_0 ]; do sleep 120; done
echo "[c68] chain67 reached seq@0 $(date +%H:%M)"
lane() {   # $1 GPU
  while true; do
    left=0; ran=0
    for j in $JOBS; do
      [ -f $A/.c68_${j/@/_} ] && continue; left=1
      ok $j || continue
      until ! busy $1 && sleep 60 && ! busy $1; do sleep 30; done
      ( set -o noclobber; : > $A/.c68_${j/@/_} ) 2>/dev/null || continue
      echo "[c68] trainer $j on $1 $(date +%H:%M)"
      CUDA_VISIBLE_DEVICES=$1 E145_SEQ=${j#*@} E147_VARS=${j%@*} E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
      ran=1; break
    done
    [ $left = 0 ] && break
    [ $ran = 0 ] && sleep 60
  done
}
lane $U0 &
sleep 40; lane $U2 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] && [ $(nev $U0) -le 1 ] && [ $(nev $U2) -le 1 ] || continue
    sleep 60   # adapter fully written
    for try in 1 2 3; do E112_PORT0=10450 E145_SEQ=${j#*@} E147_VARS=${j%@*} E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    echo "[c68] $j EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c68] ALL EXIT $(date +%m-%d\ %H:%M)"
