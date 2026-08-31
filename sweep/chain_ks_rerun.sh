#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
until [ -s results/analysis/all76_report.txt ]; do sleep 300; done
sleep 60
echo "$(date '+%m-%d %H:%M') all-76 done -> k-sweep rerun (solo)" >> results/merge_logs/chain_ks.log
.venv/bin/python sweep/ksweep_driver.py >> results/merge_logs/ksweep.log 2>&1
echo "$(date '+%m-%d %H:%M') k-sweep rerun COMPLETE" >> results/merge_logs/chain_ks.log
