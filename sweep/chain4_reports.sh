#!/bin/bash
# 10-22 reports (GPU 1 back): each report as soon as its inputs exist.
cd /home/lzhou/libero-eval; set -u
export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba E113_DEV=cuda:0
z=0; w=0; m=0; j=0; full=0
while [ $((z + w + m + j + full)) -lt 5 ]; do
  if [ $z = 0 ] && grep -q "\[g2\] SCRATCH SHARD EXIT" results/chain3_gpu2.log && grep -q "\[g1\] SCRATCH SHARD EXIT" results/chain4_gpu1.log; then
    E134_SETTING=scratch E134_STAGE=report .venv/bin/python -u sweep/exp134_stepwise.py 2>&1 | grep -v hostcfg > results/exp134z_report_console.txt; echo "[rep] SCRATCH REPORT EXIT $(date '+%m-%d %H:%M')"; z=1; fi
  if [ $w = 0 ] && grep -q "\[g2\] WP SHARD EXIT" results/chain3_gpu2.log && grep -q "\[g1\] WP SHARD EXIT" results/chain4_gpu1.log; then
    E134_SETTING=wp E134_SKIP=retain E134_TS=2,4,6,8 E134_STAGE=report .venv/bin/python -u sweep/exp134_stepwise.py 2>&1 | grep -v hostcfg > results/exp134w_even_report_console.txt; echo "[rep] WP EVEN REPORT EXIT $(date '+%m-%d %H:%M')"; w=1; fi
  if [ $m = 0 ] && grep -q "\[g1\] MVLA EXIT" results/chain4_gpu1.log; then
    E135_STAGE=report .venv/bin/python -u sweep/exp135_mergevla.py 2>&1 | grep -v hostcfg > results/exp135_report_console.txt; echo "[rep] MVLA REPORT EXIT $(date '+%m-%d %H:%M')"; m=1; fi
  if [ $j = 0 ] && grep -q "\[g0\] JOINT EVAL EXIT" results/chain4_gpu0.log; then
    E136_STAGE=report .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg > results/exp136_report_console.txt; echo "[rep] JOINT REPORT EXIT $(date '+%m-%d %H:%M')"; j=1; fi
  if [ $full = 0 ] && [ $w = 1 ] && [ "$(ls -d results/claims/*/done 2>/dev/null | wc -l)" -ge 8 ]; then
    E134_SETTING=wp E134_STAGE=report .venv/bin/python -u sweep/exp134_stepwise.py 2>&1 | grep -v hostcfg > results/exp134w_report_console.txt; echo "[rep] WP FULL REPORT EXIT $(date '+%m-%d %H:%M')"; full=1; fi
  sleep 300
done
echo "[rep] ALL REPORTS EXIT $(date '+%m-%d %H:%M')"
