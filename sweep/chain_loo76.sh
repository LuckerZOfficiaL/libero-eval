#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
# wait for the full76 pass to finish first (same CPU budget)
until [ -s results/analysis/full76_capture.json ]; do sleep 120; done
echo "$(date '+%m-%d %H:%M') strict LOO-76 starting (6 shards)" >> results/merge_logs/chain_loo76.log
for s in 0 1 2 3 4 5; do
  .venv/bin/python sweep/loo76_strict.py $s 6 > results/analysis/loo76_shard$s.log 2>&1 &
done
wait
echo "$(date '+%m-%d %H:%M') strict LOO-76 COMPLETE" >> results/merge_logs/chain_loo76.log
