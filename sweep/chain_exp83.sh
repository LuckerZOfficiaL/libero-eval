#!/bin/bash
# EXP-83 on shiva: wait for the query embeddings, validate the training loop on 40 steps, then train.
# A short validation first because the training loop is the one piece with no independent control --
# if it is wired wrong it will still produce a falling loss curve.
set -e
set -o pipefail   # without this, `cmd | tee` reports tee's exit status and a CRASH prints 'DONE'
cd /home/lzhou/libero-eval
until grep -q '^\[q\] done' results/merge_logs/exp83_queries.log 2>/dev/null; do sleep 30; done
echo "=== [e83] queries ready $(date) ==="
echo "=== [e83] VALIDATION: 40 steps $(date) ==="
.venv/bin/python sweep/exp83_train.py --steps 40 --log_every 10 --save_every 40 \
    2>&1 | tee results/merge_logs/exp83_validate.log
echo "=== [e83] VALIDATION DONE $(date) ==="
