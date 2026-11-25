#!/usr/bin/env python3
"""EXP-90 stage 2 report: per-task rank of the shared component (donor-free swap S_k + D - P_k D)."""
import json, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
from subset_probe import TARGETS
import rank_sweep as R
KS = R.KS
rows = []; L = []
P = lambda *a: L.append(" ".join(str(x) for x in a))
P("=== EXP-90: per-task rank k of the pool's shared component, donor-free swap W = S_k + D_t - P_k(D_t) (25 rollouts/cell) ===")
P(f"{'target':>7} {'base':>6} " + " ".join(f"{'k='+str(k):>6}" for k in KS) + f" {'best':>5} {'conf best':>10} {'conf k16':>9}")
for t in TARGETS:
    b, nb = R.base(t)
    sc = {k: U.get(f"e94_t{t}_k{k}", SC[t][0]) for k in KS}
    if any(v is None for v in sc.values()): continue
    kb = max(KS, key=lambda k: sc[k])
    cb = U.get(f"e94c_t{t}_k{kb}", SC[t][0]); c16 = U.get(f"e94c_t{t}_k16", SC[t][0]) if kb != 16 else cb
    rows.append(dict(t=t, b=b, sc=sc, kb=kb, cb=cb, c16=c16))
    P(f"{'t'+str(t):>7} {100*b:>5.0f}% " + " ".join(f"{100*sc[k]:>5.0f}%" for k in KS) + f" {kb:>5} "
      f"{('--' if cb is None else f'{100*cb:.0f}%'):>10} {('--' if c16 is None else f'{100*c16:.0f}%'):>9}")
n = len(rows); P(f"\nn = {n} targets")
B = np.array([r["b"] for r in rows])
for k in KS:
    v = np.array([r["sc"][k] for r in rows]); d = 100 * (v - B)
    P(f"  k={k:>2}: mean {100*v.mean():5.1f}%  vs baseline {d.mean():+5.2f}pp (SE {d.std(ddof=1)/np.sqrt(n):.2f}, p={stats.ttest_1samp(d,0).pvalue:.2f})")
M = np.array([[r["sc"][k] for k in KS] for r in rows])
# is there a per-task k effect beyond noise? two-way (target x k) ANOVA-style: variance of the k-profile within targets
resid = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
binom_var = np.mean(M * (1 - M)) / 25          # expected cell variance from 25-rollout sampling alone
P(f"target x k interaction variance {resid.var()*n*len(KS)/((n-1)*(len(KS)-1)):.4f} vs 25-rollout sampling variance {binom_var:.4f}"
  f" (ratio {resid.var()*n*len(KS)/((n-1)*(len(KS)-1))/binom_var:.2f}; ~1 = the per-task profiles are noise)")
conf = [r for r in rows if r["cb"] is not None and r["c16"] is not None and r["kb"] != 16]
if conf:
    sel = 100 * np.array([r["sc"][r["kb"]] - r["sc"][16] for r in conf])
    fr = 100 * np.array([r["cb"] - r["c16"] for r in conf])
    P(f"\nper-task best k vs k=16, {len(conf)} targets whose best k != 16:")
    P(f"  on the SELECTION rollouts: {sel.mean():+.2f}pp   on FRESH rollouts: {fr.mean():+.2f}pp (SE {fr.std(ddof=1)/np.sqrt(len(fr)) if len(fr)>1 else 0:.2f},"
      f" p={stats.ttest_1samp(fr,0).pvalue if len(fr)>1 else float('nan'):.2f}, {int((fr>0).sum())} up / {int((fr<0).sum())} down)")
    P(f"  shrinkage {100*(1 - fr.mean()/sel.mean()) if sel.mean() else float('nan'):.0f}% -> winner's curse share")
txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp90_rank_sweep_report.txt").write_text(txt + "\n")
