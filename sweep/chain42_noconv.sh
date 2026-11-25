#!/bin/bash
# 11-06 no-conversion pipeline (replaces chain41 and chain40's trainer lanes). Order 0: a plain LoRA correction trained inside
# the FROM-SCRATCH merge (consfz) = 86.9 / 77.5 raw, a 6,000-step write-protected one (consz6) = 87.5 / 77.5, cons (conversion)
# = 86.9 / 71.3. At 38 skills the converted merge collapses (library 15.6 / new 13.2) while the scratch merge holds (55.6 /
# 20.0) -> the scaling test uses consfz first. One job queue for both A6000s, in priority order:
#   f4 = consfz s4 (scaling), j4 = joint s4 (rank 64, 12k steps), f1-f3 = consfz orders 1-3, c4 = cons s4 (pre-registered arm)
# chain40's lanes are blocked by its claim files (created by hand); chain40's evaluator still evaluates cons s4 and joint s4.
# This chain evaluates consfz (port 9400; 9100 = node exporter), once each GPU has <= 2 eval workers. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c42] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
job() {   # $1 GPU, $2 job
  case $2 in
    j4) E136_ORDERS=4 E136_GPU=$1 E136_STAGE=train .venv/bin/python -u sweep/exp136_joint.py ;;
    c4) CUDA_VISIBLE_DEVICES=$1 E145_SEQ=4 E147_VARS=cons E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py ;;
    f*) CUDA_VISIBLE_DEVICES=$1 E145_SEQ=${2#f} E147_VARS=consfz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py ;;
  esac 2>&1 | grep -v hostcfg
}
lane() {   # $1 GPU
  for j in f4 j4 f1 f2 f3 c4; do
    [ -f $A/.c42_$j ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    ( set -o noclobber; : > $A/.c42_$j ) 2>/dev/null || continue
    echo "[c42] trainer $j on $1 $(date +%H:%M)"; job $1 $j
  done
}
lane $U2 &
lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=9400
left="4 1 2 3"
while [ -n "$left" ]; do
  for s in $left; do
    [ -f $A/new_loras_cons_s$s/train_consfz/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] || continue
    sleep 60   # adapter fully written
    export E145_SEQ=$s
    for try in 1 2 3; do E147_VARS=consfz E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    if [ $s = 4 ]; then rp=scr,convm,consfz; else rp=cons,consz,consfz; fi
    E147_VARS=$rp E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c42] ORDER $s EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  done
  sleep 120
done
wait; echo "[c42] ALL EXIT $(date +%m-%d\ %H:%M)"
