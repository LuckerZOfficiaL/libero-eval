#!/bin/bash
cd /home/lzhou/libero-eval
export CUDA_VISIBLE_DEVICES=""
OUT=results/analysis/hypernet_lc2.txt
echo "=== learning curve under the BINARY objective (consolidated labels) ===" > $OUT
printf "%-12s %14s %10s %8s\n" "train targets" "gain vs random" "SE" "p" >> $OUT
for N in 15 25 40 55 66; do
  .venv/bin/python sweep/hypernet.py --seeds 10 --epochs 150 --wd 1e-4 --drop 0.1 \
     --labels results/analysis/pair_labels.json --objective binary --n_train $N --tag lc2_$N 2>/dev/null \
     | grep -E "top-3 aff|random" | grep -v TRAIN > results/analysis/_lc2_$N.raw
  .venv/bin/python - "$N" <<'PY' >> $OUT
import sys, re, numpy as np
from scipy import stats
N=sys.argv[1]; h=[]; r=[]
for ln in open(f"results/analysis/_lc2_{N}.raw"):
    t=ln.strip()
    if t.startswith("hypernet") and "top-3 aff" in ln:
        h.append(float(re.search(r"top-3 aff\s+([-+][\d.]+)",ln).group(1)))
    elif t.startswith("random") and "oracle" in ln:
        r.append(float(re.findall(r"([-+][\d.]+)pp",ln)[0]))
h,r=np.array(h),np.array(r); n=min(len(h),len(r)); d=h[:n]-r[:n]
print(f"{N:<12} {d.mean():>+13.2f} {stats.sem(d):>10.2f} {stats.ttest_rel(h[:n],r[:n]).pvalue:>8.3f}")
PY
done
echo "LC2 DONE" >> $OUT
