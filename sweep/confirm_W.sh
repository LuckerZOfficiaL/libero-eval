#!/bin/bash
cd /home/lzhou/libero-eval
export CUDA_VISIBLE_DEVICES=""
for B in W WALS; do
  .venv/bin/python sweep/hypernet.py --seeds 12 --epochs 150 --wd 1e-4 --drop 0.1 \
    --blocks $B --tag conf_$B 2>/dev/null | grep -E "top-3 aff|random" | grep -v TRAIN \
    > results/analysis/_conf_$B.raw
done
.venv/bin/python - <<'PY' > results/analysis/hypernet_confirm.txt
import re, numpy as np
from scipy import stats
for B in ("W","WALS"):
    h,r,o=[],[],[]
    for ln in open(f"results/analysis/_conf_{B}.raw"):
        if ln.strip().startswith("hypernet"):
            h.append(float(re.search(r"top-3 aff\s+([-+][\d.]+)",ln).group(1)))
        elif "random" in ln:
            m=re.findall(r"([-+][\d.]+)pp",ln); r.append(float(m[0])); o.append(float(m[1]))
    h,r,o=map(np.array,(h,r,o))
    n=min(len(h),len(r)); h,r,o=h[:n],r[:n],o[:n]
    d=h-r; t=stats.ttest_rel(h,r)
    frac=(h.mean()-r.mean())/(o.mean()-r.mean())
    print(f"=== block {B}  (n={n} seeds, paired vs random on identical holdouts) ===")
    print(f"  hypernet top-3 {h.mean():+.2f}   random {r.mean():+.2f}   oracle {o.mean():+.2f}")
    print(f"  paired gain over random: {d.mean():+.2f}pp  SE {stats.sem(d):.2f}  t={t.statistic:+.2f}  p={t.pvalue:.4f}")
    print(f"  fraction of the oracle gap recovered: {100*frac:.0f}%")
    print(f"  per-seed gains: {np.round(d,1).tolist()}   positive {int((d>0).sum())}/{n}\n")
print("CONFIRM DONE")
PY
