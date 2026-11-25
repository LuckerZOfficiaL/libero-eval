#!/bin/bash
# 11-25 EXP-174 (user-approved) LoRA continual-learning baselines for the main table: O-LoRA and InfLoRA (olora / inflora) on
# 4 orders, from the library merge, no replay in training -> one run per order serves both demo budgets (InfLoRA's old-input
# subspace uses the 5-demo replay, like EWC's Fisher). Starts once chain81's 16 jobs are all claimed; lanes wait 150 s (< chain79's
# and chain80's 240 -> ahead of chain79 P2 / P3 and chain80); evals port 11500 under the eval lock. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
eval "$(grep '^JOBS=' sweep/chain81_table.sh | sed 's/^JOBS=/C81=/')"
JOBS="olora@0 inflora@0 olora@1 inflora@1 olora@2 inflora@2 olora@3 inflora@3"
echo "[c82] start $(date +%m-%d\ %H:%M): $(echo $JOBS | wc -w) jobs"
c81left() { local j; for j in $C81; do [ -f $A/.c81_${j/@/_} ] || return 0; done; return 1; }
while c81left; do sleep 300; done
echo "[c82] chain81 all claimed $(date +%m-%d\ %H:%M), lanes start"
lane2 $G_U0 c82 150 "$JOBS" &
sleep 40; lane2 $G_U2 c82 150 "$JOBS" &
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    sleep 60; evalrun $j 11500 c82 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c82] ALL EXIT $(date +%m-%d\ %H:%M)"
