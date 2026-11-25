#!/usr/bin/env python3
"""EXP-92 final: spectral truncation of FIXED donors' residuals (energy-matched) vs the standard merge.
std / top4 measured 3x (e97, e97r, e97s); top2, top8, in2 once (e97). 15 weak targets, 25 rollouts/cell."""
import pathlib, sys, numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
from subset_probe import TARGETS
import rank_sweep as R
L = []; P = lambda *a: L.append(" ".join(str(x) for x in a))
def g(p, t, a): return U.get(f"{p}_t{t}_{a}", SC[t][0])
rows = []
for t in TARGETS:
    std = [g(p, t, "std") for p in ("e97", "e97r", "e97s")]; t4 = [g(p, t, "top4") for p in ("e97", "e97r", "e97s")]
    one = {a: g("e97", t, a) for a in ("top2", "top8", "in2")}
    if None in std + t4 or None in one.values(): P(f"missing t{t}"); continue
    rows.append(dict(t=t, b=R.base(t)[0], std=np.mean(std), top4=np.mean(t4), std_runs=std, top4_runs=t4, **one))
n = len(rows); B = np.array([r["b"] for r in rows])
P(f"=== EXP-92 final: donor residual structure, fixed donors, energy-matched (n={n} weak targets) ===")
P(f"{'arm':<28}{'mean':>7}{'vs std':>10}{'p':>7}{'up/down':>9}{'vs base':>9}{'p':>7}")
S = np.array([r["std"] for r in rows])
for a, lab in (("std", "standard (3 runs, 75 rollouts)"), ("top4", "top-4 (3 runs, 75 rollouts)"), ("top2", "top-2 (1 run)"), ("top8", "top-8 (1 run)"), ("in2", "inside target subspace (1 run)")):
    v = np.array([r[a] for r in rows]); d = 100 * (v - S); db = 100 * (v - B)
    ps = stats.ttest_1samp(d, 0).pvalue if a != "std" else float("nan")
    P(f"{lab:<28}{100*v.mean():>6.1f}%{d.mean():>+9.2f}{ps:>7.3f}{int((d>0).sum()):>5}/{int((d<0).sum()):<3}{db.mean():>+9.2f}{stats.ttest_1samp(db,0).pvalue:>7.3f}")
per_run = [100 * np.mean([r["top4_runs"][i] - r["std_runs"][i] for r in rows]) for i in range(3)]
P(f"top4 - std per independent run: {', '.join(f'{x:+.2f}' for x in per_run)}")
txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp92_truncation_report.txt").write_text(txt + "\n")
