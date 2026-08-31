#!/bin/bash
# Chain: wait for the running stage-A orchestrator to finish, then launch stage B.
# Lives in tmux (session 'chain') so it survives independently of any CLI session.
cd /home/lzhou/libero-eval || exit 1
until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 120; done
sleep 15
bash sweep/launch_plan.sh sweep/goal1_wsweep_plan.json results/merge_logs/g1_wsweep.log
echo "stage B launched at $(date)" >> results/merge_logs/chain_g1c.log
