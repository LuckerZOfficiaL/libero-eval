#!/bin/bash
# 11-22 hold chain76's stream rounds until every EXP-169 (chain77) trainer is claimed (user: EXP-169 ahead of the streams; the
# 60 s vs 120 s lane delays lost a race at 05:50 because eval servers make busy2 flicker). Pre-claims chain76's unstarted jobs
# except consfzc@29 / erlc@29 (their prev @28 is held, so chain76's lanes idle instead of exiting), then releases exactly the
# claims it created once chain77 has claimed all its jobs.
cd /home/lzhou/libero-eval; set -u
. sweep/gpu_gate.sh
HOLD="consfzc@27 erlc@27 consfzc@22 erlc@22 consfzc@25 erlc@25 consfzc@28 erlc@28 consfzc@23 erlc@23 consfzc@26 erlc@26"
C77JOBS="consfzn1@0 erln1@0 consfzn1@1 erln1@1 consfzn3@0 erln3@0 consfzn3@1 erln3@1"
mine=""
for j in $HOLD; do ( set -o noclobber; : > $A/.c76_${j/@/_} ) 2>/dev/null && mine="$mine $j"; done
echo "[c76hold] $(date +%m-%d\ %H:%M) holding:$mine"
c77left() { local j; for j in $C77JOBS; do [ -f $A/.c77_${j/@/_} ] || return 0; done; return 1; }
while c77left; do sleep 60; done
for j in $mine; do rm -f $A/.c76_${j/@/_}; done
echo "[c76hold] $(date +%m-%d\ %H:%M) released:$mine"
