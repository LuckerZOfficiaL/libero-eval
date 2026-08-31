#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
wait_idle(){ until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done; }
log(){ echo "$(date '+%m-%d %H:%M') $1" >> results/merge_logs/chain_exp14.log; }
wait_idle; log "recheck done"
bash sweep/launch_plan.sh sweep/exp14_heldout_plan.json results/merge_logs/exp14_heldout.log
log "heldout launched"; sleep 60; wait_idle; log "heldout done"
bash sweep/launch_plan.sh sweep/exp14_nofam_plan.json results/merge_logs/exp14_nofam.log
log "nofam launched"; sleep 60; wait_idle; log "nofam done — EXP-14 COMPLETE"
