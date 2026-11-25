#!/bin/bash
# 11-06 EXP-147 consfz (plain LoRA, A and B trainable, trained inside the FROM-SCRATCH merge, no conversion) on orders 1-3.
# Order 0: raw 86.9 / 77.5 vs consz (WP correction, same merge) 86.9 / 61.2 and cons 86.9 / 71.3 -> does a free-A correction
# make the per-skill conversion unnecessary? Trainer lanes start only after chain40 has claimed cons s4 and joint s4 (so the
# scaling jobs keep their slots); both GPUs then split the orders via exp147's claim files. Evaluator on port 9400 (9100 =
# node exporter), once each GPU has <= 2 eval workers. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis; D=$A/new_loras_cons_s4
echo "[c41] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU
  until [ -f $D/.claim_lane_cons ] && [ -f $D/.claim_lane_joint ]; do sleep 120; done
  for s in 1 2 3; do
    [ -f $A/new_loras_cons_s$s/.claim_consfz ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    [ -f $A/new_loras_cons_s$s/.claim_consfz ] && continue
    echo "[c41] trainer s$s on $1 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E147_VARS=consfz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  done
}
lane $U2 &
lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=9400
left="1 2 3"
while [ -n "$left" ]; do
  for s in $left; do
    [ -f $A/new_loras_cons_s$s/train_consfz/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] || continue
    sleep 60   # adapter fully written
    export E145_SEQ=$s
    for try in 1 2 3; do E147_VARS=consfz E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    E147_VARS=cons,consz,consfz E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c41] ORDER $s EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  done
  sleep 120
done
wait; echo "[c41] ALL EXIT $(date +%m-%d\ %H:%M)"
