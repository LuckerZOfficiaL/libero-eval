#!/bin/bash
# 10-24 reports, restarted after the /dev/shm overflow (GPU 2's shard marker now in chain5_gpu2.log; done files only on success).
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba E113_DEV=cuda:0
w=0; m=0; full=0
while [ $((w + m + full)) -lt 3 ]; do
  if [ $w = 0 ] && grep -q "\[g2\] WP SHARD EXIT" results/chain5_gpu2.log 2>/dev/null && grep -q "\[g1\] WP SHARD EXIT" results/chain4_gpu1.log; then
    E134_SETTING=wp E134_SKIP=retain E134_TS=2,4,6,8 E134_STAGE=report .venv/bin/python -u sweep/exp134_stepwise.py 2>&1 | grep -v hostcfg > results/exp134w_even_report_console.txt; echo "[rep] WP EVEN REPORT EXIT $(date '+%m-%d %H:%M')"; w=1; fi
  if [ $m = 0 ] && grep -q "\[g1\] MVLA EXIT" results/chain4_gpu1.log; then
    E135_STAGE=report .venv/bin/python -u sweep/exp135_mergevla.py 2>&1 | grep -v hostcfg > results/exp135_report_console.txt; echo "[rep] MVLA REPORT EXIT $(date '+%m-%d %H:%M')"; m=1; fi
  if [ $full = 0 ] && [ $w = 1 ] && [ "$(ls -d results/claims/*/done 2>/dev/null | wc -l)" -ge 8 ]; then
    E134_SETTING=wp E134_STAGE=report .venv/bin/python -u sweep/exp134_stepwise.py 2>&1 | grep -v hostcfg > results/exp134w_report_console.txt; echo "[rep] WP FULL REPORT EXIT $(date '+%m-%d %H:%M')"; full=1; fi
  sleep 300
done
echo "[rep] ALL REPORTS EXIT $(date '+%m-%d %H:%M')"
