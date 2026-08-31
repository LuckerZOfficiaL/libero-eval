#!/bin/bash
cd /home/lzhou/libero-eval
export CUDA_VISIBLE_DEVICES=""
OUT=results/analysis/hypernet_v2.txt
echo "=== hypernet v2: consolidated labels (634 pairs) + binary objective ===" > $OUT
for CFG in "binary" "listwise"; do
  .venv/bin/python sweep/hypernet.py --seeds 12 --epochs 150 --wd 1e-4 --drop 0.1 \
     --labels results/analysis/pair_labels.json --objective $CFG --tag v2_$CFG 2>/dev/null \
     | grep -E "top-3 aff|random|oracle" | grep -v TRAIN > results/analysis/_v2_$CFG.raw
done
.venv/bin/python - <<'PY' >> $OUT
import re, numpy as np
from scipy import stats
for CFG in ("binary","listwise"):
    h,r,o,hh,rh=[],[],[],[],[]
    for ln in open(f"results/analysis/_v2_{CFG}.raw"):
        t=ln.strip()
        if t.startswith("hypernet") and "top-3 aff" in ln:
            h.append(float(re.search(r"top-3 aff\s+([-+][\d.]+)",ln).group(1)))
            m=re.search(r"hit (\d\.\d+) \(rand (\d\.\d+)\)",ln)
            if m: hh.append(float(m.group(1))); rh.append(float(m.group(2)))
        elif t.startswith("random") and "oracle" in ln:
            m=re.findall(r"([-+][\d.]+)pp",ln); r.append(float(m[0])); o.append(float(m[1]))
    h,r,o=map(np.array,(h,r,o)); n=min(len(h),len(r)); d=h[:n]-r[:n]
    t=stats.ttest_rel(h[:n],r[:n])
    print(f"--- objective={CFG} (n={n} seeds, paired vs random) ---")
    print(f"  top-3 gain over random: {d.mean():+.2f}pp  SE {stats.sem(d):.2f}  t={t.statistic:+.2f}  p={t.pvalue:.4f}")
    print(f"  oracle gap recovered: {100*(h.mean()-r.mean())/(o.mean()-r.mean()):.0f}%")
    if hh:
        dh=np.array(hh)-np.array(rh)
        th=stats.ttest_1samp(dh,0)
        print(f"  good-donor hit rate: model {np.mean(hh):.2f} vs random {np.mean(rh):.2f}  "
              f"(+{100*dh.mean():.0f}pp, t={th.statistic:+.2f}, p={th.pvalue:.3f})")
    print()
print("reference: prior 12-seed listwise on e27-only labels: +0.84pp, p=0.20, 10% of gap")
print("HN2 DONE")
PY
