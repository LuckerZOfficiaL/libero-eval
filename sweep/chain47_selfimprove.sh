#!/bin/bash
# 11-07 EXP-149 self-improvement round 2 (sweep/exp149_selfimprove.py) on top of consfz (4-order 87.8 / 77.5, joint 87.3 /
# 83.4). Collection = eval-like workers on the free eval slots now (2 per GPU, port 9200), orders 2 (weakest new: 67.5), 0,
# 1, 3. Round-2 trainings (1,500 steps, resumed from consfz) take each A6000 once chain44 has claimed consfzr r2 + r3;
# eval port 9250 (9100 = node exporter), <= 2 eval workers per GPU before an eval starts. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c47] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
adir() { [ $1 = 0 ] && echo $A/new_loras_cons || echo $A/new_loras_cons_s$1; }
lane() {   # $1 GPU
  until [ -f $A/.c44_r2 ] && [ -f $A/.c44_r3 ]; do sleep 120; done
  for s in 2 0 1 3; do
    [ -f $A/.c47_t$s ] && continue
    until [ -f $(adir $s)/.e149_collected ]; do sleep 120; done
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    ( set -o noclobber; : > $A/.c47_t$s ) 2>/dev/null || continue
    echo "[c47] trainer round-2 s$s on $1 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E149_STAGE=train .venv/bin/python -u sweep/exp149_selfimprove.py 2>&1 | grep -v hostcfg
  done
}
lane $U2 &
lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
for s in 2 0 1 3; do
  for try in 1 2; do E112_PER_GPU=2 E112_PORT0=9200 E145_SEQ=$s E149_STAGE=collect .venv/bin/python -u sweep/exp149_selfimprove.py 2>&1 | grep -v hostcfg; done
  : > $(adir $s)/.e149_collected; echo "[c47] collected s$s $(date +%H:%M)"
done
export E112_PER_GPU=1 E112_PORT0=9250
left="2 0 1 3"
while [ -n "$left" ]; do
  for s in $left; do
    [ -f $(adir $s)/train_consfz2/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ] || continue
    sleep 60   # adapter fully written
    export E145_SEQ=$s
    for try in 1 2 3; do E147_VARS=consfz2 E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
    E147_VARS=cons,consz,consfz,consfz2 E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    echo "[c47] ORDER $s EVAL EXIT $(date +%m-%d\ %H:%M)"
    left=$(echo $left | tr ' ' '\n' | grep -vx $s | tr '\n' ' ' | sed 's/ *$//')
  done
  sleep 120
done
wait; echo "[c47] ALL EXIT $(date +%m-%d\ %H:%M)"
