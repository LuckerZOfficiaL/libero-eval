#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done
sleep 20
bash sweep/launch_plan.sh sweep/e16_object_10_plan.json results/merge_logs/e16_object_10b.log
echo "$(date '+%m-%d %H:%M') object_10 relaunched (fixed norm stats)" >> results/merge_logs/chain_e16.log
until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done
echo "$(date '+%m-%d %H:%M') object_10 done — ALL COMBOS COMPLETE" >> results/merge_logs/chain_e16.log
