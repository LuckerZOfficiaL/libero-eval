#!/bin/bash
# EXP-40: the affinity study at 106 targets, three holdout protocols, SAME hyperparameters as the
# +1.31pp reference (run_hn2.sh: binary, 12 seeds, epochs 150, wd 1e-4, drop 0.1) and the SAME paired
# aggregation (per-seed hypernet vs random on identical holdouts). Reports pooled AND split.
cd /home/lzhou/libero-eval
test -f results/analysis/hypernet_feats106.npz || { echo "run feats106.py --stage assemble first"; exit 1; }
export CUDA_VISIBLE_DEVICES=""
HP="--objective binary --seeds 36 --epochs 150 --wd 1e-4 --drop 0.1"
F106="--labels results/analysis/pair_labels106.json --feats results/analysis/hypernet_feats106.npz --baselines results/analysis/baselines106.json"
run() {
  local tag=$1; shift
  .venv/bin/python sweep/hypernet.py $HP "$@" --tag $tag 2>/dev/null \
    | grep -E "top-3 aff|random|oracle" | grep -v TRAIN > results/analysis/_e40_$tag.raw
}
run e40_ref76  --labels results/analysis/pair_labels.json
run e40_pooled $F106 --holdout random
run e40_fleet  $F106 --holdout fleet
run e40_xdist  $F106 --holdout new
.venv/bin/python - <<'PY' | tee results/analysis/exp40_report.txt
import re, numpy as np
from scipy import stats
print("=== EXP-40: affinity study at 106 targets (874 labels) vs the 76-target reference (634) ===")
print("binary objective, 36 seeds (numpy+torch seeded per seed), epochs 150, wd 1e-4, drop 0.1; paired vs random on identical holdouts")
print("learning-curve prediction for 76 -> 106 targets: +1.31pp -> ~+1.85pp (slope 0.018pp/target)\n")
DESC = {"e40_ref76":  "D. CONTROL   76 targets, holdout 5 weak+5 strong (the reference protocol)",
        "e40_pooled": "A. POOLED    106 targets, holdout 5 weak+5 strong from all 106",
        "e40_fleet":  "B. FLEET-HO  106 targets, holdout drawn from fleet only (new targets always train)",
        "e40_xdist":  "C. X-DIST    train on the 76 fleet targets, test on ALL 30 new (goal/object/spatial)"}
for tag in ("e40_ref76", "e40_pooled", "e40_fleet", "e40_xdist"):
    h, r, o = [], [], []
    for ln in open(f"results/analysis/_e40_{tag}.raw"):
        t = ln.strip()
        if t.startswith("hypernet") and "top-3 aff" in ln:
            h.append(float(re.search(r"top-3 aff\s+([-+][\d.]+)", ln).group(1)))
        elif t.startswith("random") and "oracle" in ln:
            m = re.findall(r"([-+][\d.]+)pp", ln); r.append(float(m[0])); o.append(float(m[1]))
    h, r, o = map(np.array, (h, r, o)); n = min(len(h), len(r)); d = h[:n] - r[:n]
    print(DESC[tag])
    if n == 0:
        print("  (no output)\n"); continue
    tt = stats.ttest_rel(h[:n], r[:n]) if n > 1 else None
    gap = 100 * (h.mean() - r.mean()) / max(o.mean() - r.mean(), 1e-9)
    print(f"  top-3 gain over random: {d.mean():+.2f}pp  SE {stats.sem(d) if n>1 else 0:.2f}"
          + (f"  t={tt.statistic:+.2f}  p={tt.pvalue:.4f}" if tt else "")
          + f"   random {r.mean():+.2f}  oracle {o.mean():+.2f}  gap recovered {gap:.0f}%   (n={n} seeds)\n")
PY
echo "EXP40 DONE"
