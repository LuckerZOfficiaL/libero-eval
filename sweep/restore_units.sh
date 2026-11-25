#!/bin/bash
# 10-22: restored parts of the per-step plan, as 8 claimable units (atomic mkdir claim; any free GPU takes the next one).
# Caller exports the GPU env (CUDA_VISIBLE_DEVICES, ORCH_*_MAP, E113_GPUS, E113_DEV, E112_PORT0, E113_ROUND) and TAG.
# 10-24: `done` only on exit 0; a failed unit releases its claim and is retried on a later pass (10-24 /dev/shm overflow
# marked 6 crashed units done).
cd /home/lzhou/libero-eval; set -u; mkdir -p results/claims
ALLBUT_RETAIN=ours_w0.25,opcm,avg,ours_w0.1,ours_budget
for pass in 1 2 3; do for k in 0 1 2 3; do for kind in odd retain; do
  u=wp_${kind}_${k}
  mkdir results/claims/$u 2>/dev/null || continue
  echo "[$TAG] claim $u $(date '+%m-%d %H:%M')"
  if [ $kind = odd ]; then E134_SETTING=wp E134_SKIP=retain E134_TS=1,3,5,7 E134_SHARD=$k/4 .venv/bin/python -u sweep/exp134_stepwise.py
  else E134_SETTING=wp E134_SKIP=$ALLBUT_RETAIN E134_TS=1,2,3,4,5,6,7,8 E134_SHARD=$k/4 .venv/bin/python -u sweep/exp134_stepwise.py; fi
  if [ $? = 0 ]; then touch results/claims/$u/done; echo "[$TAG] done $u $(date '+%m-%d %H:%M')"
  else rm -rf results/claims/$u; echo "[$TAG] FAILED $u (released) $(date '+%m-%d %H:%M')"; sleep 600; fi
done; done; done
echo "[$TAG] RESTORE LOOP EXIT $(date '+%m-%d %H:%M')"
