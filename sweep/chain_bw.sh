#!/bin/bash
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null || tmux ls 2>/dev/null | grep -q "^lw:"; do sleep 120; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
.venv/bin/python sweep/backweight_driver.py
