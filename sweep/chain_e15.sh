#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
wait_idle(){ until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done; }
log(){ echo "$(date '+%m-%d %H:%M') $1" >> results/merge_logs/chain_e15b.log; }
for s in e0 e1 e2; do
  bash sweep/launch_plan.sh sweep/e15b_${s}_plan.json results/merge_logs/e15b_${s}.log
  log "$s launched"; sleep 60; wait_idle; log "$s done"
done
log "EXP-15 E0+E1+E2 COMPLETE"
