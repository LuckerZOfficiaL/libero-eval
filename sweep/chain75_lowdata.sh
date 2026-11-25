#!/bin/bash
# 11-21 EXP-166 low-data case for our method (user: "find a clear case for our method to lead consistently"): (B) the continual
# stream at 1 library demo per skill, ours (consfzd1c) vs ER (erld1c), E145_SEQ 19 / 20 / 4, rounds resumed from consfzd1@0 /
# erld1@0; (A) ours vs ER at 1 demo on orders 1-3 (consfzd1 / erld1). Stream rounds first (they chain), then the orders.
# Nothing else queued: lanes wait 90 s; evals port 10750 under the eval lock.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
JOBS="consfzd1c@19 erld1c@19 consfzd1c@20 erld1c@20 consfzd1c@4 erld1c@4 consfzd1@1 erld1@1 consfzd1@2 erld1@2 consfzd1@3 erld1@3"
echo "[c75] start $(date +%m-%d\ %H:%M)"
lane2 $G_U0 c75 90 "$JOBS" &
sleep 40; lane2 $G_U2 c75 90 "$JOBS" &
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    sleep 60; evalrun $j 10750 c75 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c75] ALL EXIT $(date +%m-%d\ %H:%M)"
