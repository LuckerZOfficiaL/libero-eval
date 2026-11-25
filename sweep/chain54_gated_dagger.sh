#!/bin/bash
# 11-08 EXP-156 gated DAgger (sweep/exp155_dagger.py, arm consfzg): orders 0, 1, 3 (the pre-registered test), then 2 (the gate
# was chosen on its test results). Relabel helpers: one per A6000, gated skills only, claims shared with odin's chain50.
# Training lanes: each A6000 takes the next relabeled order ahead of chain53's f5 / j5 (polls every 30 s and claims after 60 s
# free; chain53's lanes wait 180 s). Eval port 9600, 1 worker per GPU, <= 2 eval workers per GPU before it starts. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c54] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
adir() { [ $1 = 0 ] && echo $A/new_loras_cons || echo $A/new_loras_cons_s$1; }
ready() { E145_SEQ=$1 E155_STAGE=ready .venv/bin/python sweep/exp155_dagger.py >/dev/null 2>&1; }
helper() {   # $1 GPU
  for s in 0 1 3; do CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E155_STAGE=relabel .venv/bin/python -u sweep/exp155_dagger.py 2>&1 | grep --line-buffered -v -e hostcfg -e NVML; done
  echo "[c54] helper on $1 done $(date +%H:%M)"
}
lane() {   # $1 GPU
  for s in 0 1 3 2; do
    [ -f $A/.c54_g$s ] && continue
    until ready $s; do sleep 120; done
    until ! busy $1 && sleep 60 && ! busy $1; do sleep 30; done
    ( set -o noclobber; : > $A/.c54_g$s ) 2>/dev/null || continue
    echo "[c54] trainer g$s on $1 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E155_STAGE=train .venv/bin/python -u sweep/exp155_dagger.py 2>&1 | grep -v hostcfg
  done
}
helper $U0 &
sleep 20; helper $U2 &
lane $U0 &
sleep 30; lane $U2 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 E112_PORT0=9600 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
left="0 1 3 2"
while [ -n "$left" ]; do
  for s in $left; do
    [ -f $(adir $s)/train_consfzg/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] || continue
    sleep 60   # adapter fully written
    export E145_SEQ=$s
    for try in 1 2 3; do E147_VARS=consfzg E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    E147_VARS=cons,consfz,consfzd,consfzg E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c54] ORDER $s EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  done
  sleep 120
done
wait; echo "[c54] ALL EXIT $(date +%m-%d\ %H:%M)"
