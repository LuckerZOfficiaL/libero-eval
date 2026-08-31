#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 180; done
sleep 30
echo "$(date '+%m-%d %H:%M') pending jobs done -> Phase A starting" >> results/merge_logs/chain_phaseA.log
.venv/bin/python sweep/loo_decomposition.py > results/analysis/loo_decomposition_report.txt 2>&1
echo "$(date '+%m-%d %H:%M') Phase A COMPLETE" >> results/merge_logs/chain_phaseA.log
