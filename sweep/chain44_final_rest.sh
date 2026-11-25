#!/bin/bash
# 11-07 consfzr (no conversion, no demos) orders 1-3, deferred behind chain43's f2 / f3 / c4: consfzr order 0 = 78.8 / 65.0 vs
# consfz 86.9 / 77.5 and consr 90.6 / 71.2, and consfz order 1 < cons -> whether conversion matters (f2 / f3) and whether the
# conversion pipeline scales (c4) come first. chain43's r1-r3 claims were made by hand; these lanes start once chain43 has
# claimed c4 and use their own claims. chain43's evaluator (port 9300) evaluates the adapters. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c44] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
lane() {   # $1 GPU
  until [ -f $A/.c43_c4 ]; do sleep 120; done
  for j in r1 r2 r3; do
    [ -f $A/.c44_$j ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    ( set -o noclobber; : > $A/.c44_$j ) 2>/dev/null || continue
    echo "[c44] trainer $j on $1 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 E145_SEQ=${j#r} E147_VARS=consfzr E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  done
}
lane $U2 &
lane $U0 &
wait; echo "[c44] ALL EXIT $(date +%m-%d\ %H:%M)"
