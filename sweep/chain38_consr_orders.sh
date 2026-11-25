#!/bin/bash
# 11-06 EXP-147 consr (no demos at all) on orders 1-3 (order 0: raw 90.6 / 71.2 vs cons 86.9 / 71.3). Library rollouts from
# chain34. Two trainer lanes (GPU2 after consh, GPU0 after consz6) split the orders via exp147's claim files; one evaluator
# takes each order as its adapter lands, once each GPU has <= 2 eval workers; port 9700 (9100 = node exporter). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c38] start $(date +%m-%d\ %H:%M)"
busy() { for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU, $2 adapter that must exist first (that GPU's current job)
  until [ -f $A/new_loras_cons/$2/adapter_model.safetensors ]; do sleep 120; done
  for s in 1 2 3; do
    [ -f $A/new_loras_cons_s$s/.claim_consr ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    [ -f $A/new_loras_cons_s$s/.claim_consr ] && continue
    echo "[c38] trainer s$s on $1 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E147_VARS=consr E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  done
}
lane $U2 train_consh &
lane $U0 train_consz6 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=9700
left="1 2 3"
while [ -n "$left" ]; do
  for s in $left; do
    [ -f $A/new_loras_cons_s$s/train_consr/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] || continue
    sleep 60   # adapter fully written
    export E145_SEQ=$s
    E147_VARS=consr E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    for try in 1 2 3; do E147_VARS=consr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    E147_VARS=cons,consz,consr E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c38] ORDER $s EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  done
  sleep 120
done
wait; echo "[c38] ALL EXIT $(date +%m-%d\ %H:%M)"
