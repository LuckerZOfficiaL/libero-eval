#!/bin/bash
cd /home/lzhou/libero-eval
export CUDA_VISIBLE_DEVICES=""
OUT=results/analysis/hypernet_cand_curve.txt
echo "=== Does MORE CANDIDATES PER TARGET help? (targets fixed at 66) ===" > $OUT
printf "%-12s %14s %10s %8s\n" "candidates" "gain vs random" "SE" "p" >> $OUT
for C in 3 5 8; do
  .venv/bin/python sweep/hypernet.py --seeds 10 --epochs 150 --wd 1e-4 --drop 0.1 \
     --blocks WALS --n_cand $C --tag cc_$C 2>/dev/null \
     | grep -E "top-3 aff|random" | grep -v TRAIN > results/analysis/_cc_$C.raw
  .venv/bin/python - "$C" <<'PY' >> $OUT
import sys, re, numpy as np
from scipy import stats
C=sys.argv[1]; h=[]; r=[]
for ln in open(f"results/analysis/_cc_{C}.raw"):
    s=ln.strip()
    if s.startswith("hypernet") and "top-3 aff" in ln:
        h.append(float(re.search(r"top-3 aff\s+([-+][\d.]+)",ln).group(1)))
    elif s.startswith("random") and "oracle" in ln:
        r.append(float(re.findall(r"([-+][\d.]+)pp",ln)[0]))
h,r=np.array(h),np.array(r); n=min(len(h),len(r)); d=h[:n]-r[:n]
print(f"{C:<12} {d.mean():>+13.2f} {stats.sem(d):>10.2f} {stats.ttest_rel(h[:n],r[:n]).pvalue:>8.3f}")
PY
done
echo "CC DONE" >> $OUT
