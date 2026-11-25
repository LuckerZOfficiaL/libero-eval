#!/bin/bash
set -e; set -o pipefail
cd /home/lzhou/libero-eval
for B in P WA; do
  echo "=== [e85ls] blocks=$B START $(date) ==="
  .venv/bin/python sweep/exp45_deploy.py --stage labelspace --feats results/analysis/hypernet_feats_fp.npz --blocks $B --seeds 36 2>&1 | grep --line-buffered -v hostcfg | tee results/analysis/exp85_labelspace_$B.txt
  echo "=== [e85ls] blocks=$B OK $(date) ==="
done
echo "=== [e85ls] ALL DONE $(date) ==="
