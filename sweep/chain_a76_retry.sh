#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
until [ -s results/analysis/ksweep_report.txt ]; do sleep 300; done
sleep 60
rm -f results/analysis/all76_report.txt
echo "$(date '+%m-%d %H:%M') k-sweep done -> all-76 retry of missing targets (solo)" >> results/merge_logs/chain_a76.log
.venv/bin/python sweep/all76_driver.py >> results/merge_logs/all76.log 2>&1
echo "$(date '+%m-%d %H:%M') all-76 retry COMPLETE" >> results/merge_logs/chain_a76.log
