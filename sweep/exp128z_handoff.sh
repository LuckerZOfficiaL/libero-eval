#!/bin/bash
# 10-20: odin's single P6000 is too slow for the scratch RETAIN run (~30 episodes/h). When shiva frees up (EXP-131 report),
# stop the odin run, copy its finished records to shiva, finish the remainder on shiva's 3 GPUs, then mark RUN EXIT locally
# so the existing watcher syncs + reports on shiva.
cd /home/lzhou/libero-eval
until ssh shiva 'grep -q "REPORT EXIT" /home/lzhou/libero-eval/results/exp131.log 2>/dev/null'; do sleep 300; done
echo "[handoff] shiva free $(date +%H:%M); stopping odin run"
tmux kill-session -t e128z_odin 2>/dev/null; sleep 30
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
rsync -a results/merge/e112_c128z*.jsonl shiva:/home/lzhou/libero-eval/results/merge/
ssh shiva 'cd /home/lzhou/libero-eval && rm -rf /dev/shm/mergesweep/cache /dev/shm/mergesweep/prebuilt/e12* /dev/shm/mergesweep/prebuilt/e13*; E128_ADAPTERS=scratch E113_GPUS=0,1,2 E113_DEV=cuda:1 E113_ROUND=3 .venv/bin/python -u sweep/exp128_retain.py 2>&1 | tail -5'
rsync -a shiva:/home/lzhou/libero-eval/results/merge/e112_c128z*.jsonl results/merge/
echo "[e128z] RUN EXIT (finished on shiva) $(date +%H:%M)" >> results/exp128z.log
