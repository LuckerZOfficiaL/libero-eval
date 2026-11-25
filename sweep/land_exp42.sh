#!/bin/bash
# EXP-42 landing: labels -> baselines -> features (DIM_A 24 and 48) -> affinity study with variants as
# TRAINING-ONLY targets, evaluated on EXP-40's unchanged holdouts. One command; every stage cached.
cd /home/lzhou/libero-eval
set -e
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python sweep/labels_ext.py
PLAN=sweep/plan_ext.json TAG=ext .venv/bin/python - <<'PY'
import sys, json; sys.path.insert(0, "sweep")
import feats106 as F
b = F.baselines(); json.dump({str(k): v for k, v in b.items()}, open("results/analysis/baselines_ext.json", "w"), indent=0)
print(f"baselines_ext.json: {len(b)} targets ({sum(1 for t in b if t >= 106)} variants)")
PY
for DA in 24 48; do
  PLAN=sweep/plan_ext.json TAG=ext FTAG=exta$DA DIMA=$DA .venv/bin/python sweep/feats106.py --stage assemble > results/analysis/feats_exta$DA.log 2>&1
  tail -1 results/analysis/feats_exta$DA.log
done
HP="--objective binary --seeds 36 --epochs 150 --wd 1e-4 --drop 0.1 --blocks WA"
run() { local tag=$1; shift; .venv/bin/python sweep/hypernet.py $HP "$@" --tag $tag 2>/dev/null | grep -E "top-3 aff|random|oracle" | grep -v TRAIN > results/analysis/_e42_$tag.raw; }
for DA in 24 48; do for HO in fleet new; do
  run ext_a${DA}_$HO --labels results/analysis/pair_labels_ext.json --feats results/analysis/hypernet_featsexta$DA.npz --baselines results/analysis/baselines_ext.json --holdout $HO
  run r106_a${DA}_$HO --labels results/analysis/pair_labels106.json --feats results/analysis/hypernet_feats106a$DA.npz --baselines results/analysis/baselines106.json --holdout $HO
done; done
.venv/bin/python - <<'PY' | tee results/analysis/exp42_report.txt
import re, numpy as np, json
from scipy import stats
def gain(f):
    h, r = [], []
    for ln in open(f):
        t = ln.strip()
        if t.startswith("hypernet") and "top-3 aff" in ln and "TRAIN" not in ln: h.append(float(re.search(r"top-3 aff\s+([-+][\d.]+)", ln).group(1)))
        elif t.startswith("random") and "oracle" in ln: r.append(float(re.findall(r"([-+][\d.]+)pp", ln)[0]))
    n = min(len(h), len(r)); d = np.array(h[:n]) - np.array(r[:n])
    return d.mean(), stats.sem(d), stats.ttest_1samp(d, 0).pvalue
L = json.load(open("results/analysis/pair_labels_ext.json")); nt = len({k.split("_")[0] for k in L}); nv = len({k.split("_")[0] for k in L if int(k.split("_")[0]) >= 106})
print(f"=== EXP-42: manufactured targets as extra TRAINING data — {len(L)} labels, {nt} targets ({nv} seed variants, train-only) ===")
print("W+A model, seeded 36 seeds; holdouts identical to EXP-40 (variants never held out). Gain over random, pp.\n")
print(f"{'training set':<26}{'DIM_A':>6}{'B fleet-holdout':>22}{'C cross-distribution':>26}")
for DA in (24, 48):
    for name, tag in (("106 targets (reference)", f"r106_a{DA}"), (f"106 + {nv} variants", f"ext_a{DA}")):
        row = f"{name:<26}{DA:>6}"
        for HO in ("fleet", "new"):
            g, se, p = gain(f"results/analysis/_e42_{tag}_{HO}.raw"); row += f"{g:>+9.2f} ±{se:.2f} p={p:.3f}".rjust(22 if HO == "fleet" else 26)
        print(row)
print(f"\nslope reference: 76 -> 106 real tasks gave +0.52pp on B (0.017pp/target); {nv} variants at that slope predict +{0.017*nv:.2f}pp")
PY
echo "EXP42 LANDED"
