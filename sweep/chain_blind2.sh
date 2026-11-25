#!/bin/bash
# Odin queue v3 (09-14 17:10): EXP-80 weak slice is decided, so the cheap EXP-32 completion goes BEFORE the
# strong tail (which cannot finish by the 09-16 meeting anyway). EXP-80 and EXP-78 resume idempotently.
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 30; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null
echo "=== [blind2] 1/3 EXP-32 completion $(date) ==="; .venv/bin/python sweep/unified_pool_driver.py
echo "=== [blind2] 2/3 EXP-80 strong tail resumes $(date) ==="; .venv/bin/python sweep/blind_pool.py
echo "=== [blind2] 3/3 EXP-78 resumes $(date) ==="; .venv/bin/python sweep/subset_probe.py
echo "=== [blind2] ALL DONE $(date) ==="
