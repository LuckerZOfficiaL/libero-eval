#!/bin/bash
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null || tmux ls 2>/dev/null | grep -q "^sym"; do
  sleep 120
done
.venv/bin/python sweep/dosing_driver.py
