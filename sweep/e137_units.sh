#!/bin/bash
# 10-24 EXP-137 as 3 claimable shards (atomic mkdir in results/claims137/; done only on exit 0; a failed shard is released
# and retried on a later pass). Caller exports the GPU env (CUDA_VISIBLE_DEVICES, ORCH_*_MAP, E113_GPUS, E113_DEV, E112_PORT0)
# and TAG.
cd /home/lzhou/libero-eval; set -u; mkdir -p results/claims137
for pass in 1 2 3; do for k in 0 1 2; do
  u=e137_$k
  mkdir results/claims137/$u 2>/dev/null || continue
  echo "[$TAG] claim $u $(date '+%m-%d %H:%M')"
  E137_SHARD=$k/3 E113_ROUND=1 .venv/bin/python -u sweep/exp137_dilution.py
  if [ $? = 0 ]; then touch results/claims137/$u/done; echo "[$TAG] done $u $(date '+%m-%d %H:%M')"
  else rm -rf results/claims137/$u; echo "[$TAG] FAILED $u (released) $(date '+%m-%d %H:%M')"; sleep 600; fi
done; done
echo "[$TAG] E137 LOOP EXIT $(date '+%m-%d %H:%M')"
