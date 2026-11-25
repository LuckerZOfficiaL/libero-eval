#!/bin/bash
# SHIVA queue once the machine is returned (user, 09-14): sync -> EXP-77 remainder -> EXP-81 OOD robustness.
# Run ON SHIVA. Bracketed pgrep, in a script file.
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 60; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
echo "=== [shiva] 1/2 EXP-77 remainder (3 veryweak S16-init targets) $(date) ==="
bash sweep/queue_shiva17.sh
echo "=== [shiva] 2/2 EXP-81 OOD robustness $(date) ==="
.venv/bin/python sweep/ood_robust.py
echo "=== [shiva] ALL DONE $(date) ==="
