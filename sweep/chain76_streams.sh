#!/bin/bash
# 11-21 EXP-167 the continual stream on orders 1-3 (5 library demos, 1,500 resumed steps per arrival), ours (consfzc) vs ER
# (erlc): E145_SEQ 21-23 / 24-26 / 27-29 = order 1 / 2 / 3's library with 14 / 18 / 22 new; arrival 1 = consfz@o / erl@o (done).
# Lanes wait 120 s so chain75 (90 s) keeps priority; evals port 10800 once chain75 has no pending eval.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
C75JOBS="consfzd1c@19 erld1c@19 consfzd1c@20 erld1c@20 consfzd1c@4 erld1c@4 consfzd1@1 erld1@1 consfzd1@2 erld1@2 consfzd1@3 erld1@3"
JOBS=""; for a in 0 1 2; do for o in 21 24 27; do for v in consfzc erlc; do JOBS="$JOBS $v@$((o + a))"; done; done; done; JOBS=${JOBS# }
echo "[c76] start $(date +%m-%d\ %H:%M): $JOBS"
lane2 $G_U0 c76 120 "$JOBS" &
sleep 40; lane2 $G_U2 c76 120 "$JOBS" &
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    until ! pending "$C75JOBS"; do sleep 120; done
    sleep 60; evalrun $j 10800 c76 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c76] ALL EXIT $(date +%m-%d\ %H:%M)"
