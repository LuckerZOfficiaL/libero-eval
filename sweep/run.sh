#!/bin/bash
# One command to run a merge experiment. Generates the plan, then launches it in tmux 'merge'.
#
#   bash sweep/run.sh                                 # defaults: task1+task2, task_arithmetic, 10 trials
#   bash sweep/run.sh --pairs 3:4 --method ties       # override via flags
#   bash sweep/run.sh --scope 5 --method average      # five pairs
#   bash sweep/run.sh --config configs/merge.yaml     # or set everything in a YAML file
#
# Results per pair+method: results/merge/task_<a>_<b>_<method>.jsonl
cd /home/lzhou/libero-eval
PLAN=sweep/merge_plan.json
# Default to configs/merge.yaml unless you pass your own --config. Any flags you add override the
# YAML, so `bash sweep/run.sh` runs exactly what's in merge.yaml; `bash sweep/run.sh --pairs 3:4`
# runs merge.yaml with pairs overridden.
case " $* " in
  *" --config "*) CFG=() ;;
  *) CFG=(--config configs/merge.yaml) ;;
esac
.venv/bin/python sweep/make_merge_plan.py "${CFG[@]}" "$@" --out "$PLAN" || exit 1
# stop any previous merge run and free the GPUs (bracket patterns avoid self-match)
pkill -f 'merge_[o]rchestrator.py' 2>/dev/null; sleep 1
pkill -f 'serve_[p]olicy' 2>/dev/null; pkill -f 'libero/[m]ain.py' 2>/dev/null; sleep 2
rm -rf /dev/shm/mergesweep/w*
mkdir -p results/merge results/merge_logs results/merge_videos
tmux kill-session -t merge 2>/dev/null
tmux new-session -d -s merge ".venv/bin/python sweep/merge_orchestrator.py $PLAN 2>&1 | tee results/merge_logs/orchestrator.log"
sleep 3
echo ""
echo "launched in tmux 'merge'."
echo "  check   : bash sweep/merge_health.sh"
echo "  summary : python sweep/merge_summary.py"
echo "  attach  : tmux attach -t merge"
