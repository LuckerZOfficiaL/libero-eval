#!/bin/bash
# Odin queue, reordered 09-13 on user request: the provenance cross-product comes first.
#  1. EXP-79  external donors -> LIBERO-90 receivers   (the missing cell of the cross-product)
#  2. unionfix EXP-32 completion: x_goal_expert_4 + f28's two probes
#  3. EXP-78  subset probing, resumed (idempotent: its 53 finished probes are kept)
cd /home/lzhou/libero-eval
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
echo "=== [xprod] 1/3 EXP-79 external donors -> fleet receivers $(date) ==="
.venv/bin/python sweep/extdonor_fleet.py
echo "=== [xprod] 2/3 EXP-32 completion $(date) ==="
.venv/bin/python sweep/unified_pool_driver.py
echo "=== [xprod] 3/3 EXP-78 subset probing resumed $(date) ==="
.venv/bin/python sweep/subset_probe.py
echo "=== [xprod] ALL THREE DONE $(date) ==="
