#!/bin/bash
# 11-20 EXP-164d: what the library-merge start is worth in training steps. consb6 / consb12 = base start, rank 16, same data as
# consb / erl (order 0), 6,000 / 12,000 steps; compare with ER from the merge at 3,000 (erl@0 84.4 / 72.5). Replaces consb64@0
# (rank 64 + 12k steps confounded steps with capacity; pre-claimed in chain72 so its lanes skip it). Trainer lanes wait 120 s
# (chain72 waits 90 s -> its queue keeps priority); evals wait for no pending eval of chains 67b / 68b / 70b / 72, port 10650.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
C72EV="consfzq@0 erlq@0 consfzq@1 erlq@1 consfaq@0 consfaq@1 consfz10@0 erl10@0 consfz10@1 erl10@1 consbc@19 consbc@20 consbc@4 erlr@0 consbr@0 erld1@0 consbd1@0 erld25@0 consbd25@0"
JOBS="consb6@0 consb12@0"
echo "[c73] start $(date +%m-%d\ %H:%M)"
lane2 $G_U0 c73 120 "$JOBS" &
sleep 40; lane2 $G_U2 c73 120 "$JOBS" &
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    until ! pending "$C67JOBS $C68JOBS $C70JOBS $C72EV"; do sleep 120; done
    sleep 60; evalrun $j 10650 c73 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c73] ALL EXIT $(date +%m-%d\ %H:%M)"
