#!/bin/bash
# 11-19 replaces chain67's eval loop. chain67's lanes are idle (its trainer jobs moved to chain71) but its eval loop ignored the
# gap evals: 04:31 its sca@4 ref and chain69c's rae@0 started together -> servers died during load (rae@0 21/24). Every eval on
# shiva now holds results/analysis/.evlock (gpu_gate evalrun). This chain waits for chain67's sca@4 eval to exit, stops chain67
# (no trainer inside), completes sca@4 if tasks are missing, then evaluates chain67's remaining jobs as chain71 trains them
# (2 workers per GPU as chain67, port 10350; gap chains still yield to these through gapwait's pending check).
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
echo "[c67b] start $(date +%m-%d\ %H:%M)"
exec 8> $A/.evlock; flock 8   # hold the eval lock while chain67's own (unlocked) sca@4 eval finishes
until grep -q "sca@4 EVAL EXIT" results/chain67.log; do sleep 10; done
tmux kill-session -t c67 && echo "[c67b] chain67 stopped $(date +%H:%M)"
exec 8>&-
evalrun sca@4 10400 c67b 2
left=$(for j in $C67JOBS; do grep -q "\] $j EVAL EXIT" results/chain67.log || echo $j; done | tr '\n' ' ' | sed 's/ *$//')
echo "[c67b] to evaluate: $left"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    sleep 60   # adapter fully written
    evalrun $j 10350 c67b 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
echo "[c67b] ALL EXIT $(date +%m-%d\ %H:%M)"
