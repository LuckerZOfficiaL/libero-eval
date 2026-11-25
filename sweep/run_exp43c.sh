#!/bin/bash
# EXP-43c: capacity of the activation block (the cross-suite signal). Rebuild features with DIMA in
# {12,24,48,96} (W fixed at 24), then WA-only models, seeded 36 seeds, both holdouts.
cd /home/lzhou/libero-eval
export CUDA_VISIBLE_DEVICES=""
HP="--objective binary --seeds 36 --epochs 150 --wd 1e-4 --drop 0.1"
for DA in 12 24 48 96; do
  DIMA=$DA FTAG=106a$DA .venv/bin/python sweep/feats106.py --stage assemble > results/analysis/feats106a$DA.log 2>&1
  for HO in fleet new; do
    .venv/bin/python sweep/hypernet.py $HP --labels results/analysis/pair_labels106.json --feats results/analysis/hypernet_feats106a$DA.npz \
      --baselines results/analysis/baselines106.json --holdout $HO --blocks WA --tag e43c_${HO}_a$DA 2>/dev/null \
      | grep -E "top-3 aff|random|oracle" | grep -v TRAIN > results/analysis/_e43c_${HO}_a$DA.raw
  done
done
.venv/bin/python - <<'PY' | tee results/analysis/exp43c_report.txt
import re, numpy as np
from scipy import stats
def gain(f):
    h, r = [], []
    for ln in open(f):
        t = ln.strip()
        if t.startswith("hypernet") and "top-3 aff" in ln and "TRAIN" not in ln: h.append(float(re.search(r"top-3 aff\s+([-+][\d.]+)", ln).group(1)))
        elif t.startswith("random") and "oracle" in ln: r.append(float(re.findall(r"([-+][\d.]+)pp", ln)[0]))
    n = min(len(h), len(r)); d = np.array(h[:n]) - np.array(r[:n])
    return d.mean(), stats.sem(d), stats.ttest_1samp(d, 0).pvalue
print("=== EXP-43c: activation-block capacity, W+A model, 106 targets, 36 seeds (gain over random, pp) ===")
print(f"{'DIM_A':<8}{'B fleet-holdout':>22}{'C cross-distribution':>26}")
for DA in (12, 24, 48, 96):
    row = f"{DA:<8}"
    for HO in ("fleet", "new"):
        g, se, p = gain(f"results/analysis/_e43c_{HO}_a{DA}.raw"); row += f"{g:>+9.2f} ±{se:.2f} p={p:.3f}".rjust(22 if HO == "fleet" else 26)
    print(row)
print("\nreference: WA at DIM_A=24 from 43a: +0.53 / +0.83;  all blocks: +0.72 / +0.82")
PY
echo "EXP43c DONE"
