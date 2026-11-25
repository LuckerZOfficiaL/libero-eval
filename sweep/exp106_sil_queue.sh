#!/bin/bash
# EXP-106: train self-imitation adapters for a list of targets sequentially on one GPU.
# usage: exp106_sil_queue.sh DEVICE T1,T2,...   (skips targets whose adapter already exists)
cd /home/lzhou/libero-eval
for t in ${2//,/ }; do
  if [ -f results/analysis/exp106_sil/t$t/sil_meta.json ]; then echo "[silq] t$t exists"; continue; fi
  .venv/bin/python -u sweep/exp106_sil.py $t --device $1 2>&1 | grep --line-buffered -vE "hostcfg|Warning"
done
echo "[silq] EXIT"
