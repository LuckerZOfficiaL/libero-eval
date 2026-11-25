#!/bin/bash
# EXP-43a: which feature block carries the transfer at 106 targets? Seeded protocol (36 seeds), reference
# hyperparameters, two holdouts: B fleet-only (new targets train) and C cross-distribution (test = 30 new).
cd /home/lzhou/libero-eval
export CUDA_VISIBLE_DEVICES=""
HP="--objective binary --seeds 36 --epochs 150 --wd 1e-4 --drop 0.1"
F106="--labels results/analysis/pair_labels106.json --feats results/analysis/hypernet_feats106.npz --baselines results/analysis/baselines106.json"
for HO in fleet new; do for B in WALS W A L S WA WS; do
  .venv/bin/python sweep/hypernet.py $HP $F106 --holdout $HO --blocks $B --tag e43_${HO}_${B} 2>/dev/null \
    | grep -E "top-3 aff|random|oracle" | grep -v TRAIN > results/analysis/_e43_${HO}_${B}.raw
done; done
.venv/bin/python - <<'PY' | tee results/analysis/exp43_blocks_report.txt
import re, numpy as np
from scipy import stats
def gain(f):
    h, r = [], []
    for ln in open(f):
        t = ln.strip()
        if t.startswith("hypernet") and "top-3 aff" in ln and "TRAIN" not in ln: h.append(float(re.search(r"top-3 aff\s+([-+][\d.]+)", ln).group(1)))
        elif t.startswith("random") and "oracle" in ln: r.append(float(re.findall(r"([-+][\d.]+)pp", ln)[0]))
    n = min(len(h), len(r)); d = np.array(h[:n]) - np.array(r[:n])
    return d.mean(), stats.sem(d), stats.ttest_1samp(d, 0).pvalue, n
print("=== EXP-43a: feature-block ablation at 106 targets (seeded, 36 seeds; gain over random, pp) ===")
print("blocks: W weight-Gram row | A activation signature | L language tf-idf | S ranks+baseline\n")
print(f"{'blocks':<8}{'B fleet-holdout':>22}{'C cross-distribution':>26}")
for B in ("WALS", "W", "A", "L", "S", "WA", "WS"):
    row = f"{B:<8}"
    for HO in ("fleet", "new"):
        g, se, p, n = gain(f"results/analysis/_e43_{HO}_{B}.raw")
        row += f"{g:>+9.2f} ±{se:.2f} p={p:.3f}".rjust(22 if HO == "fleet" else 26)
    print(row)
print("\nreference (EXP-40, all blocks): B +0.72 ±0.23   C +0.82 ±0.08")
PY
echo "EXP43a DONE"
