#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done
sleep 20
bash sweep/launch_plan.sh sweep/e15c_ps_plan.json results/merge_logs/e15c_ps.log
echo "$(date '+%m-%d %H:%M') per-suite control launched" >> results/merge_logs/chain_e15b.log
until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done
echo "$(date '+%m-%d %H:%M') per-suite control done — E2 2x2 COMPLETE" >> results/merge_logs/chain_e15b.log
