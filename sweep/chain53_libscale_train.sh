#!/bin/bash
# 11-07 EXP-150 trainings, relaunched: chain48's first f6 / j6 attempts crashed at data loading (LIVING_ROOM_SCENE2 butter ->
# basket has no regenerated no-noops demos; train_lora now falls back to the original LIBERO demos, no-ops dropped). chain48's
# lanes stopped by PID; its evaluator (ports 9450 / 9500) still evaluates whatever adapter appears. Each A6000 takes the next
# of j6 (longest first), f6, f5, j5 when free; claims .c53_j6 / .c53_f6 and chain48's .c48_f5 / .c48_j5 (chain49 waits on
# those). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c53] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
lane() {   # $1 GPU
  for j in j6 f6 f5 j5; do
    c=$A/.c53_$j; [ $j = f5 -o $j = j5 ] && c=$A/.c48_$j
    [ -f $c ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    ( set -o noclobber; : > $c ) 2>/dev/null || continue
    s=${j#?}; echo "[c53] trainer $j on $1 $(date +%H:%M)"
    if [ ${j:0:1} = f ]; then CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E147_VARS=consfz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
    else E136_ORDERS=$s E136_GPU=$1 E136_STAGE=train .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg; fi
  done
}
lane $U2 &
sleep 30
lane $U0 &
wait; echo "[c53] ALL EXIT $(date +%m-%d\ %H:%M)"
