#!/bin/bash
# 11-07 the candidate final pipeline, consfzr = training-free merge of the from-scratch adapters + ONE plain LoRA trained inside
# it on the new skills' own rollouts + the library's own rollouts (no conversion, no demos at all), orders 0-3 + the 38-skill
# scaling order 4. Evidence so far: consfz (no conversion, library demos) order 0 86.9 / 77.5, order 4 88.1 / 81.8; consr
# (conversion, no demos) 4 orders 89.4 / 81.9. Takes over chain42's remaining queue (its f2 / f3 / c4 claims made by hand;
# chain42's evaluator still evaluates consfz, chain40's cons s4 + joint s4). Queue, both A6000s:
#   r0 r4 r1 r2 r3 (consfzr) > f2 f3 (consfz orders 2-3) > c4 (cons s4). Eval port 9300 (9100 = node exporter). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c43] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
adir() { [ $1 = 0 ] && echo $A/new_loras_cons || echo $A/new_loras_cons_s$1; }
job() {   # $1 GPU, $2 job
  case $2 in
    c4) CUDA_VISIBLE_DEVICES=$1 E145_SEQ=4 E147_VARS=cons E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py ;;
    f*) CUDA_VISIBLE_DEVICES=$1 E145_SEQ=${2#f} E147_VARS=consfz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py ;;
    r*) CUDA_VISIBLE_DEVICES=$1 E145_SEQ=${2#r} E147_VARS=consfzr E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py ;;
  esac 2>&1 | grep -v hostcfg
}
lane() {   # $1 GPU
  for j in r0 r4 r1 r2 r3 f2 f3 c4; do
    [ -f $A/.c43_$j ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    ( set -o noclobber; : > $A/.c43_$j ) 2>/dev/null || continue
    echo "[c43] trainer $j on $1 $(date +%H:%M)"; job $1 $j
  done
}
lane $U2 &
lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=9300
left="0 4 1 2 3"
while [ -n "$left" ]; do
  for s in $left; do
    [ -f $(adir $s)/train_consfzr/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] || continue
    sleep 60   # adapter fully written
    export E145_SEQ=$s
    for try in 1 2 3; do E147_VARS=consfzr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    if [ $s = 4 ]; then rp=scr,convm,consfz,consfzr; else rp=cons,consr,consz,consfz,consfzr; fi
    E147_VARS=$rp E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c43] ORDER $s EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  done
  sleep 120
done
wait; echo "[c43] ALL EXIT $(date +%m-%d\ %H:%M)"
