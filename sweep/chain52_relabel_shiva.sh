#!/bin/bash
# 11-07 EXP-155 relabeling helpers for order 2 on shiva's A6000s (odin's P6000 alone: ~5 h per order), one per GPU next to
# the running trainer + eval worker; skills claimed one at a time (shared with chain50 on odin). GPU1 idle.
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c52] start $(date +%m-%d\ %H:%M)"
for g in $U0 $U2; do CUDA_VISIBLE_DEVICES=$g E145_SEQ=2 E155_STAGE=relabel .venv/bin/python -u sweep/exp155_dagger.py 2>&1 | grep --line-buffered -v -e hostcfg -e NVML & sleep 30; done
wait; echo "[c52] RELABELED s2 $(date +%m-%d\ %H:%M)"
