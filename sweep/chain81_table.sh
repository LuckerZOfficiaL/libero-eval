#!/bin/bash
# 11-24 EXP-173 (user-approved) the all-baselines table: every same-data baseline on 4 orders at 5 and 1 library demos.
# Fills: multi-task LoRA from the base at 1 demo (consbd1 @1-3), sequential fine-tuning and online EWC (seq / ewc @1-3; no
# replay in training -> one run per order serves both budgets; EWC's Fisher uses the 5-demo replay), learned merge weights
# without correction (lrc @1-3 at 5 demos, lrcd1 @0-3 at 1 demo). Order-complete batches first.
# Starts once chain79's seed block (P1) is all claimed; lanes wait 150 s (< chain79's 240 -> ahead of its P2 / P3);
# evals port 11400 under the eval lock. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
eval "$(grep '^P1=' sweep/chain79_consolidate.sh)"
JOBS="consbd1@1 lrcd1@0 seq@1 ewc@1 lrcd1@1 lrc@1 consbd1@2 seq@2 ewc@2 lrcd1@2 lrc@2 consbd1@3 seq@3 ewc@3 lrcd1@3 lrc@3"
echo "[c81] start $(date +%m-%d\ %H:%M): $(echo $JOBS | wc -w) jobs"
p1left() { local j; for j in $P1; do [ -f $A/.c79_${j/@/_} ] || return 0; done; return 1; }
while p1left; do sleep 300; done
echo "[c81] chain79 seed block all claimed $(date +%m-%d\ %H:%M), lanes start"
lane2 $G_U0 c81 150 "$JOBS" &
sleep 40; lane2 $G_U2 c81 150 "$JOBS" &
left="$JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    sleep 60; evalrun $j 11400 c81 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c81] ALL EXIT $(date +%m-%d\ %H:%M)"
