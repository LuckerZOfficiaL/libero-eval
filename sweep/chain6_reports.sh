#!/bin/bash
# 10-24 EXP-137 report when its 3 shards are done.
cd /home/lzhou/libero-eval; set -u
until [ "$(ls -d results/claims137/*/done 2>/dev/null | wc -l)" -ge 3 ]; do sleep 300; done
CUDA_VISIBLE_DEVICES="" E137_STAGE=report .venv/bin/python -u sweep/exp137_dilution.py 2>&1 | grep -v hostcfg > results/exp137_report_console.txt
echo "[rep] E137 REPORT EXIT $(date '+%m-%d %H:%M')"
