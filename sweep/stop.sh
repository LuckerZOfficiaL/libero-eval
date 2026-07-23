#!/bin/bash
# Stop any running merge experiment cleanly and free the GPUs/shm.
# Usage: bash sweep/stop.sh
# (Bracketed patterns avoid the pkill matching its own shell.)
pkill -f 'merge_[o]rchestrator.py' 2>/dev/null
pkill -f 'merge_[l]oras.py'        2>/dev/null   # merge builds
pkill -f 'serve_[p]olicy'          2>/dev/null   # policy servers
pkill -f 'libero/[m]ain.py'        2>/dev/null   # eval clients
sleep 2
rm -rf /dev/shm/mergesweep/w*
tmux kill-session -t merge 2>/dev/null
echo "stopped."
echo "  orchestrator: $(pgrep -f 'merge_[o]rchestrator.py' | wc -l)   servers: $(pgrep -f 'serve_[p]olicy' | wc -l)   clients: $(pgrep -f 'libero/[m]ain.py' | wc -l)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "note: completed (pair,scene) results are kept in results/merge/ (a re-run resumes / skips them)."
