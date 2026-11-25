#!/bin/bash
set -o pipefail
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null; do sleep 30; done
rm -rf /dev/shm/mergesweep/cache/* 2>/dev/null || true
echo "=== [gs3] strong 55 @10, heavy_parallel=2 START $(date) ==="
GT_TARGETS=0,1,2,3,4,5,7,10,11,12,13,16,17,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,57,58,59,61,64,65,66,69,70 GT_N=10 GT_AHEAD=24 GT_HEAVY=2 .venv/bin/python sweep/gt_stream.py
GT_TARGETS=0,1,2,3,4,5,7,10,11,12,13,16,17,21,22,23,24,25,26,27,28,29,30,31,32,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,57,58,59,61,64,65,66,69,70 GT_N=10 GT_HEAVY=2 CHUNK=7 .venv/bin/python sweep/gt_sweep.py
echo "=== [gs3] EXIT $? $(date) ==="
