#!/bin/bash
# tmux-launch an orchestrator run for a given plan. Exists as a SEPARATE script so that
# watcher/chain scripts never contain the orchestrator's literal path in their own command
# line — a `pgrep -f 'merge_orchestrat[o]r.py'` inside a script whose cmdline embeds the
# unbracketed literal matches ITSELF and deadlocks (this bit us on 2026-07-28).
# Usage: bash sweep/launch_plan.sh <plan.json> <logfile>
PLAN=$1
LOG=$2
cd /home/lzhou/libero-eval || exit 1
tmux kill-session -t merge 2>/dev/null
tmux new-session -d -s merge ".venv/bin/python sweep/merge_orchestrator.py $PLAN 2>&1 | tee $LOG"
echo "launched: $PLAN (log: $LOG)"
