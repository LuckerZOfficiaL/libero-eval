#!/bin/bash
# 11-22 EXP-171 (user-approved) consolidation of the framework, after chain78. Priority order:
#  1 seeds: seed-1 replicates of the claims in the abstract (1-demo ours vs ER on 4 orders; merge vs base start, ER from the
#    library merge (erls1) and from the base (consbs1) on 4 orders; the order-0 stream for ours / ER / base, 19 -> 20 -> 4)
#  2 held-out orders 2-3 for the budget cells chosen on orders 0-1: 750 steps (consfzq / consfaq / erlq), 1 successful rollout
#    (consfzn1 / erln1), 25 library demos (consfzd25 / erld25), the base start at 12,000 steps (consb12)
#  3 breadth: 5th arrival = all 30 externals on the 16- (consfzc / erlc @34) and 32-skill (consfzl / erll @33) libraries; the
#    32-skill library on order 1 (consfz6 / erl6 @35, stream consfzl / erll @36-38)
# Starts once every chain78 trainer (+ its joint) is claimed and c79col (rollouts of the 8 new externals) is done; lanes wait
# 240 s; evals port 11000 under the eval lock. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
C78JOBS="erl6@5 consb6@5 consfzl@30 erll@30 consfzl@31 erll@31 consfzl@32 erll@32 joint@32"
P1="consfzd1s1@0 erld1s1@0 consfzd1s1@1 erld1s1@1 consfzd1s1@2 erld1s1@2 consfzd1s1@3 erld1s1@3"
P1="$P1 consfzs1@0 erls1@0 consbs1@0 consfzcs1@19 erlcs1@19 consbcs1@19 consfzcs1@20 erlcs1@20 consbcs1@20 consfzcs1@4 erlcs1@4 consbcs1@4"
P1="$P1 erls1@1 consbs1@1 erls1@2 consbs1@2 erls1@3 consbs1@3"
P2="consfzq@2 consfaq@2 erlq@2 consfzq@3 consfaq@3 erlq@3 consfzn1@2 erln1@2 consfzn1@3 erln1@3"
P2="$P2 consfzd25@2 erld25@2 consfzd25@3 erld25@3 consb12@2 consb12@3"
P3="consfzc@34 erlc@34 consfzl@33 erll@33 consfz6@35 erl6@35 consfzl@36 erll@36 consfzl@37 erll@37 consfzl@38 erll@38"
JOBS="$P1 $P2 $P3"
echo "[c79] start $(date +%m-%d\ %H:%M): $(echo $JOBS | wc -w) jobs"
c78left() { local j; for j in $C78JOBS; do [ -f $A/.c78_${j/@/_} ] || return 0; done; return 1; }
while c78left || ! grep -q "c79col\] ALL EXIT" results/c79col.log 2>/dev/null; do sleep 300; done
echo "[c79] chain78 trainers all claimed, rollouts collected $(date +%m-%d\ %H:%M), lanes start"
lane2 $G_U0 c79 240 "$JOBS" &
sleep 40; lane2 $G_U2 c79 240 "$JOBS" &
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    sleep 60; evalrun $j 11000 c79 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c79] ALL EXIT $(date +%m-%d\ %H:%M)"
