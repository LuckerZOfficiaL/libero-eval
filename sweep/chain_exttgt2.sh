#!/bin/bash
# EXP-27 rerun with per-expert norm stats, behind whatever holds the orchestrator.
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 90; done
.venv/bin/python sweep/external_target_driver.py
