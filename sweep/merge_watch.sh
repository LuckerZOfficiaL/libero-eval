#!/bin/bash
# Live auto-refreshing merge health. Usage: bash sweep/merge_watch.sh [plan.json] [interval_seconds]
# Ctrl-C to stop.
PLAN=${1:-sweep/merge_plan.json}
INT=${2:-60}
trap 'echo; echo "stopped watching."; exit 0' INT
while true; do
  clear
  bash /home/lzhou/libero-eval/sweep/merge_health.sh "$PLAN"
  echo
  echo "(refreshing every ${INT}s — Ctrl-C to stop)"
  sleep "$INT"
done
