#!/usr/bin/env python3
"""EXP-89 stage 2 report: local shared component swap (nn8 / rnd8) vs pool S16 swap (e94 k16) vs baseline."""
import pathlib, sys, numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
from subset_probe import TARGETS
import rank_sweep as R
L = []; P = lambda *a: L.append(" ".join(str(x) for x in a))
P("=== EXP-89: swap the shared component for a LOCAL one, donor-free (25 rollouts/cell) ===")
P(f"{'target':>7} {'base':>6} {'pool S16':>9} {'nn8':>6} {'rnd8':>6}")
rows = []
for t in TARGETS:
    b, _ = R.base(t)
    p16 = [x for x in (U.get(f"e94_t{t}_k16", SC[t][0]), U.get(f"e94c_t{t}_k16", SC[t][0])) if x is not None]
    nn, rn = U.get(f"e95_t{t}_nn8", SC[t][0]), U.get(f"e95_t{t}_rnd8", SC[t][0])
    if not p16 or nn is None or rn is None: continue
    rows.append((t, b, float(np.mean(p16)), nn, rn))
    P(f"{'t'+str(t):>7} {100*b:>5.0f}% {100*np.mean(p16):>8.0f}% {100*nn:>5.0f}% {100*rn:>5.0f}%")
A = np.array([r[1:] for r in rows]); n = len(A)
def pr(nm, a, b):
    d = 100 * (a - b); return f"{nm}: {d.mean():+.2f}pp (SE {d.std(ddof=1)/np.sqrt(n):.2f}, p={stats.ttest_rel(a, b).pvalue:.2f}, {int((d>0).sum())} up / {int((d<0).sum())} down)"
P(f"\nn = {n}   means: base {100*A[:,0].mean():.1f}  pool {100*A[:,1].mean():.1f}  nn8 {100*A[:,2].mean():.1f}  rnd8 {100*A[:,3].mean():.1f}")
for nm, i, j in (("nn8 - pool", 2, 1), ("nn8 - rnd8", 2, 3), ("rnd8 - pool", 3, 1), ("nn8 - base", 2, 0), ("pool - base", 1, 0)):
    P("  " + pr(nm, A[:, i], A[:, j]))
txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp89_local_shared_report.txt").write_text(txt + "\n")
