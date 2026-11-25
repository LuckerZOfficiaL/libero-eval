#!/bin/bash
# 11-21 EXP-169 (user-approved): ours (consfzn*) vs ER (erln*) on the first 1 / 3 successful rollouts per new skill, orders 0 / 1,
# 3,000 steps, 5 library demos. Priority (user): after chain75's 1-demo runs, ahead of chain76's streams -> the lanes start once
# every chain75 trainer is claimed, then wait 60 s (chain76 waits 120 s). Evals port 10850 under the eval lock.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
C75JOBS="consfzd1c@19 erld1c@19 consfzd1c@20 erld1c@20 consfzd1c@4 erld1c@4 consfzd1@1 erld1@1 consfzd1@2 erld1@2 consfzd1@3 erld1@3"
JOBS="consfzn1@0 erln1@0 consfzn1@1 erln1@1 consfzn3@0 erln3@0 consfzn3@1 erln3@1"
echo "[c77] start $(date +%m-%d\ %H:%M): $JOBS"
c75left() { local j; for j in $C75JOBS; do [ -f $A/.c75_${j/@/_} ] || return 0; done; return 1; }
while c75left; do sleep 120; done
echo "[c77] chain75 trainers all claimed $(date +%m-%d\ %H:%M), lanes start"
lane2 $G_U0 c77 60 "$JOBS" &
sleep 40; lane2 $G_U2 c77 60 "$JOBS" &
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    sleep 60; evalrun $j 10850 c77 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c77] ALL EXIT $(date +%m-%d\ %H:%M)"
