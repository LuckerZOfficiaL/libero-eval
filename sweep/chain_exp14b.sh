#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
wait_idle(){ until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done; }
log(){ echo "$(date '+%m-%d %H:%M') $1" >> results/merge_logs/chain_exp14b.log; }
for stage in recheck heldout nofam; do
  bash sweep/launch_plan.sh sweep/exp14_${stage}_plan.json results/merge_logs/exp14_${stage}2.log
  log "$stage launched (GPU1 only)"; sleep 60; wait_idle; log "$stage done"
done
log "EXP-14 RERUN COMPLETE"
