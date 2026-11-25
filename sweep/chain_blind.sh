#!/bin/bash
# Odin queue, reordered 09-13 00:40 after the user's design correction.
#  1. EXP-79  external donors -> fleet receivers (resumes; ~5h)         -> slide 3
#  2. EXP-80  provenance-BLIND union, 8 uniform candidates, weak first  -> replaces slide 1
#  3. unionfix EXP-32 completion (x_goal_expert_4 + f28)
#  4. EXP-78  subset probing resumes
cd /home/lzhou/libero-eval
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
echo "=== [blindchain] 1/4 EXP-79 $(date) ==="; .venv/bin/python sweep/extdonor_fleet.py
echo "=== [blindchain] 2/4 EXP-80 $(date) ==="; .venv/bin/python sweep/blind_pool.py
echo "=== [blindchain] 3/4 EXP-32 completion $(date) ==="; .venv/bin/python sweep/unified_pool_driver.py
echo "=== [blindchain] 4/4 EXP-78 resume $(date) ==="; .venv/bin/python sweep/subset_probe.py
echo "=== [blindchain] ALL DONE $(date) ==="
