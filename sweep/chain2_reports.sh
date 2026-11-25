#!/bin/bash
# 10-21 plan: run each report as soon as its inputs exist (both shards / its stage), write REPORT markers.
export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba E113_DEV=cuda:0
cd /home/lzhou/libero-eval; set -u
done_wp=0; done_mv=0; done_j=0; done_z=0
while [ $((done_wp + done_mv + done_j + done_z)) -lt 4 ]; do
  if [ $done_wp = 0 ] && grep -q "\[g0\] WP SHARD EXIT" results/chain2_gpu0.log && grep -q "\[g2\] WP SHARD EXIT" results/chain2_gpu2.log; then
    E134_SETTING=wp E134_SKIP=retain E134_STAGE=report .venv/bin/python -u sweep/exp134_stepwise.py 2>&1 | grep -v hostcfg > results/exp134w_report_console.txt; echo "[rep] WP REPORT EXIT $(date '+%m-%d %H:%M')"; done_wp=1; fi
  if [ $done_mv = 0 ] && grep -q "\[g2\] MVLA EXIT" results/chain2_gpu2.log; then
    E135_STAGE=report .venv/bin/python -u sweep/exp135_mergevla.py 2>&1 | grep -v hostcfg > results/exp135_report_console.txt; echo "[rep] MVLA REPORT EXIT $(date '+%m-%d %H:%M')"; done_mv=1; fi
  if [ $done_j = 0 ] && grep -q "\[g0\] JOINT EVAL EXIT" results/chain2_gpu0.log; then
    E136_STAGE=report .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg > results/exp136_report_console.txt; echo "[rep] JOINT REPORT EXIT $(date '+%m-%d %H:%M')"; done_j=1; fi
  if [ $done_z = 0 ] && grep -q "\[g0\] SCRATCH SHARD EXIT" results/chain2_gpu0.log && grep -q "\[g2\] SCRATCH SHARD EXIT" results/chain2_gpu2.log; then
    E134_SETTING=scratch E134_TS=2,4,6,8 E134_STAGE=report .venv/bin/python -u sweep/exp134_stepwise.py 2>&1 | grep -v hostcfg > results/exp134z_report_console.txt; echo "[rep] SCRATCH REPORT EXIT $(date '+%m-%d %H:%M')"; done_z=1; fi
  sleep 300
done
echo "[rep] ALL REPORTS EXIT $(date '+%m-%d %H:%M')"
