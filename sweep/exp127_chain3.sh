#!/bin/bash
# EXP-127 three-GPU continuation (10-19, GPU 0 free again): GPU 0 runs the merges whose adapters already exist (1 model per
# round, separate ports) until training ends; then base on GPUs 1,2; then the full run on GPUs 0,1,2; then report.
cd /home/lzhou/libero-eval
set -u
rm -f results/e127_stop
( E113_GPUS=0 E113_DEV=cuda:0 E113_ROUND=1 E112_PORT0=8100 E127_AVAILABLE_ONLY=1 E127_STOP_FILE=results/e127_stop \
    .venv/bin/python -u sweep/exp127_replicate_wp.py 2>&1 | tee -a results/exp127_g0.log; echo "[e127] G0 EXIT $(date +%H:%M)" ) &
until [ "$(grep -c 'training shard done' results/exp127.log)" -ge 2 ]; do sleep 60; done
echo "[e127] TRAIN EXIT $(date +%H:%M)"
touch results/e127_stop
E113_GPUS=1,2 E113_DEV=cuda:1 E113_ROUND=3 E127_STAGE=base .venv/bin/python -u sweep/exp127_replicate_wp.py
echo "[e127] BASE EXIT $(date +%H:%M)"
wait
rm -rf /dev/shm/mergesweep/cache /dev/shm/mergesweep/prebuilt/e127_*
E113_GPUS=0,1,2 E113_DEV=cuda:1 E113_ROUND=3 E127_STAGE=run .venv/bin/python -u sweep/exp127_replicate_wp.py
echo "[e127] RUN EXIT $(date +%H:%M)"
E127_STAGE=report .venv/bin/python -u sweep/exp127_replicate_wp.py 2>&1 | grep -v hostcfg | tee results/exp127_report_console.txt
echo "[e127] REPORT EXIT $(date +%H:%M)"
