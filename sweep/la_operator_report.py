#!/usr/bin/env python3
"""EXP-92 / EXP-93 report (la_operator_arms.py)."""
import pathlib, sys, numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
from subset_probe import TARGETS
import rank_sweep as R
L = []; P = lambda *a: L.append(" ".join(str(x) for x in a))
def table(title, arms, ref_name, ref_fn, prefix):
    P(f"\n=== {title} (25 rollouts/cell) ===")
    P(f"{'target':>7} {'base':>6} {ref_name:>8} " + " ".join(f"{a:>6}" for a in arms))
    rows = []
    for t in TARGETS:
        b, _ = R.base(t); ref = ref_fn(t); v = [U.get(f"{prefix}_t{t}_{a}", SC[t][0]) for a in arms]
        if ref is None or any(x is None for x in v): continue
        rows.append([b, ref] + v)
        P(f"{'t'+str(t):>7} {100*b:>5.0f}% {100*ref:>7.0f}% " + " ".join(f"{100*x:>5.0f}%" for x in v))
    A = np.array(rows); n = len(A)
    P(f"n = {n}; means: base {100*A[:,0].mean():.1f}, {ref_name} {100*A[:,1].mean():.1f}, " + ", ".join(f"{a} {100*A[:,2+i].mean():.1f}" for i, a in enumerate(arms)))
    for i, a in enumerate(arms):
        for j, nm in ((1, ref_name), (0, "base")):
            d = 100 * (A[:, 2+i] - A[:, j])
            P(f"  {a} - {nm}: {d.mean():+.2f}pp (SE {d.std(ddof=1)/np.sqrt(n):.2f}, p={stats.ttest_1samp(d,0).pvalue:.2f}, {int((d>0).sum())} up / {int((d<0).sum())} down)")
    return A
def pool(t):
    v = [x for x in (U.get(f"e94_t{t}_k16", SC[t][0]), U.get(f"e94c_t{t}_k16", SC[t][0])) if x is not None]
    return float(np.mean(v)) if v else None
table("EXP-93: own-residual dose, no donors: W = S16 + beta * r_t", ["b75", "b125", "b150"], "beta=1", pool, "e96")
A = table("EXP-92: fixed donors (probe-committed top-3), donor residuals transformed, energy-matched", ["in2", "top4"], "std",
          lambda t: U.get(f"e97_t{t}_std", SC[t][0]), "e97")
d = 100 * (A[:, 1] - A[:, 0]); P(f"  std (the method's commit, fresh) - base: {d.mean():+.2f}pp (SE {d.std(ddof=1)/np.sqrt(len(d)):.2f}, p={stats.ttest_1samp(d,0).pvalue:.3f}, {int((d>0).sum())} up / {int((d<0).sum())} down)")
txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp92_93_report.txt").write_text(txt + "\n")
