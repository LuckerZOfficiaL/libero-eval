#!/bin/bash
# 11-20 EXP-164b for our method: consfzd1 / consfzd25 = ours (merge of library + arrivals, correction) at 1 / 25 library demos
# per skill, order 0, next to ER's erld1 / erld25 (ER merges the library once; ours re-merges with each arrival). Trainer lanes
# wait 105 s (chain72 90 s keeps priority; chain73's step sweep, 120 s, runs after); evals port 10700 after chains 67b-73's.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
C72EV="consfzq@0 erlq@0 consfzq@1 erlq@1 consfaq@0 consfaq@1 consfz10@0 erl10@0 consfz10@1 erl10@1 consbc@19 consbc@20 consbc@4 erlr@0 consbr@0 erld1@0 consbd1@0 erld25@0 consbd25@0"
JOBS="consfzd1@0 consfzd25@0"
echo "[c74] start $(date +%m-%d\ %H:%M)"
lane2 $G_U0 c74 105 "$JOBS" &
sleep 40; lane2 $G_U2 c74 105 "$JOBS" &
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    until ! pending "$C67JOBS $C68JOBS $C70JOBS $C72EV"; do sleep 120; done
    sleep 60; evalrun $j 10700 c74 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c74] ALL EXIT $(date +%m-%d\ %H:%M)"
