#!/bin/bash
# EXP-33 on SHIVA, strictly after the REINFORCE run (one orchestrator at a time).
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null || tmux ls 2>/dev/null | grep -q "^rl:"; do sleep 120; done
.venv/bin/python sweep/layerwise_probe.py
