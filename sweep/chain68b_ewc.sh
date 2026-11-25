#!/bin/bash
# 11-18 v2 of chain68_ewc.sh (stopped before it started: its lanes / gap evals race chain67, see gpu_gate.sh). EXP-159 online
# EWC in the arrival stream: ewc@0 (lambda 1e-3 pre-registered) -> ewcc@19 / 20 / 4 (resumed, previous ewc.pt). Starts once
# chain67 has claimed seq@0; trainer lanes yield to chain67 (90 s double check + GPU reservation); gap evals 1 worker per GPU,
# port 10450, only when chain67 has no pending eval and both A6000s have >= 16 GB free. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
echo "[c68] start $(date +%m-%d\ %H:%M)"
until [ -f $A/.c67_seq_0 ]; do sleep 120; done
echo "[c68] chain67 reached seq@0 $(date +%H:%M)"
lane2 $G_U0 c68 90 "$C68JOBS" &
sleep 40; lane2 $G_U2 c68 90 "$C68JOBS" &
left="$C68JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    gapwait 90 "$C67JOBS"; evalrun $j 10450 c68
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c68] ALL EXIT $(date +%m-%d\ %H:%M)"
