#!/bin/bash
cd /home/lzhou/libero-eval
export CUDA_VISIBLE_DEVICES=""
OUT=results/analysis/hypernet_ablation.txt
echo "=== E4 feature-block ablation (4 seeds, holdout = 10 unseen targets) ===" > $OUT
echo "W=residual-Gram  A=activation sig  L=language  S=scalars" >> $OUT
printf "%-6s %16s %10s %10s\n" block "top-3 aff" "top-1" spearman >> $OUT
for B in W A L S WA WL WALS; do
  R=$(.venv/bin/python sweep/hypernet.py --seeds 4 --epochs 150 --wd 1e-4 --drop 0.1 \
        --blocks $B --tag abl_$B 2>/dev/null | grep -E "^hypernet|^random|^oracle")
  H=$(echo "$R" | grep '^hypernet'); RD=$(echo "$R" | grep '^random'); OR=$(echo "$R" | grep '^oracle')
  printf "%-6s %s\n" "$B" "$(echo $H | cut -d' ' -f2-)" >> $OUT
  echo "        (random $(echo $RD | cut -d' ' -f2)  oracle $(echo $OR | cut -d' ' -f2))" >> $OUT
done
echo "ABLATION DONE" >> $OUT
