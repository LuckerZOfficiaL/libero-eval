#!/bin/bash
# 11-08 EXP-156 order 3 released (order 1 = +13.7 new >= the +8.7 futility bar): gated DAgger g3 on the first A6000 free,
# ahead of chain57's 6,000-step s6 run (claims after 20 s free vs 60 s). chain54's lanes keep skipping g3 (.c54_g3 stays as
# the hold); chain54's evaluator (port 9600) evaluates the adapter. Order 2 stays held. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c58] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
lane() {   # $1 GPU
  until ! busy $1 && sleep 20 && ! busy $1; do sleep 10; done
  ( set -o noclobber; : > $A/.c58_g3 ) 2>/dev/null || return
  echo "[c58] trainer g3 on $1 $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES=$1 E145_SEQ=3 E155_STAGE=train .venv/bin/python -u sweep/exp155_dagger.py 2>&1 | grep -v hostcfg
}
lane $U0 &
sleep 5; lane $U2 &
wait; echo "[c58] ALL EXIT $(date +%m-%d\ %H:%M)"
