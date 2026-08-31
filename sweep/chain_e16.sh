#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
wait_idle(){ until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done; }
log(){ echo "$(date '+%m-%d %H:%M') $1" >> results/merge_logs/chain_e16.log; }
for p in e16_object_10 e16_10_goal e17_3way; do
  bash sweep/launch_plan.sh sweep/${p}_plan.json results/merge_logs/${p}.log
  log "$p launched"; sleep 60; wait_idle; log "$p done"
done
log "ALL COMBO EXPERIMENTS COMPLETE"
