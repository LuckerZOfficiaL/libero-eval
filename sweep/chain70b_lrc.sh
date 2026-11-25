#!/bin/bash
# 11-18 v2 of chain70_lrc.sh: chain70's trainer lrcfz@0 (pid $1) keeps running inside chain70 (its tmux pane is the trainer's
# controlling terminal); once it exits, chain70 is stopped (before its race-prone eval loop starts, >= 60 s after the adapter
# appears) and this chain evaluates lrcfz@0 and trains + evaluates lrc@0 (claim .c70b_lrc_0; chain70's own lrc@0 claim was
# pre-created so its lanes exit). Lanes yield to chain67 and 68b (120 s double check + reservation); gap evals port 10550, after
# chain67 / 68b's pending evals.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
echo "[c70b] start $(date +%m-%d\ %H:%M), waiting for chain70's trainer (pid $1)"
while kill -0 $1 2>/dev/null; do sleep 10; done
tmux kill-session -t c70 2>/dev/null && echo "[c70b] chain70 stopped $(date +%H:%M)"
lane2 $G_U0 c70b 120 "lrc@0" &
sleep 40; lane2 $G_U2 c70b 120 "lrc@0" &
left="$C70JOBS"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    gapwait 120 "$C67JOBS $C68JOBS"; evalrun $j 10550 c70b
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c70b] ALL EXIT $(date +%m-%d\ %H:%M)"
