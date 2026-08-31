#!/bin/bash
# Fill the one cell lost to a transient safetensors I/O error, AFTER the symmetry run finishes.
# Strict one-orchestrator-at-a-time: wait for no orchestrator AND no sym session.
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null || tmux ls 2>/dev/null | grep -q "^sym"; do
  sleep 120
done
.venv/bin/python sweep/dosing_driver.py
