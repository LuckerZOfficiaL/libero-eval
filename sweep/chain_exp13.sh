#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
wait_idle(){ until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done; }
log(){ echo "$(date '+%H:%M') $1" >> results/merge_logs/chain_exp13.log; }
bash sweep/launch_plan.sh sweep/exp13_a_plan.json results/merge_logs/exp13_a.log
log "stage a launched"; sleep 60; wait_idle; log "stage a done"
.venv/bin/python sweep/exp13_make_bc.py >> results/merge_logs/chain_exp13.log 2>&1 || { log "plan-gen FAILED"; exit 1; }
bash sweep/launch_plan.sh sweep/exp13_b_plan.json results/merge_logs/exp13_b.log
log "stage b launched"; sleep 60; wait_idle; log "stage b done"
bash sweep/launch_plan.sh sweep/exp13_c_plan.json results/merge_logs/exp13_c.log
log "stage c launched"; sleep 60; wait_idle; log "stage c done — EXP-13 COMPLETE"
