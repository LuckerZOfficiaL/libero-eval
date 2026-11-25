#!/usr/bin/env python3
"""EXP-78: subset probe vs singleton probe, paired on target x candidate-menu. Abstention scores baseline."""
import sys, numpy as np
sys.path.insert(0, 'sweep'); sys.path.insert(0, 'scripts')
from scipy import stats
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
from subset_probe import TARGETS, DRAWS, subsets
from draw_variance import cands
rows = []
for k in DRAWS:
    for t in TARGETS:
        task = SC[t][0]; b = target_baseline(t)
        sub = {j: U.get(f"e86_t{t}_d{k}_s{j}", task) for j in range(8)}; sub = {a: v for a, v in sub.items() if v is not None}
        sing = {d: U.get(f"e75_t{t}_d{k}_p{d}", task) for d in cands(t, k)}; sing = {a: v for a, v in sing.items() if v is not None}
        if len(sub) < 6 or len(sing) < 6: continue
        cS = U.get(f"e87_t{t}_d{k}", task); cP = U.get(f"e76_t{t}_d{k}", task)
        abS = max(sub.values()) <= b; abP = max(sing.values()) <= b
        eS = b if abS else cS; eP = b if abP else cP
        if eS is None or eP is None: continue
        rows.append((t, k, b, eS, eP, abS, abP))
gS = np.array([(r[3]-r[2])*100 for r in rows]); gP = np.array([(r[4]-r[2])*100 for r in rows]); d = gS - gP
se = lambda a: a.std(ddof=1)/np.sqrt(len(a))
print(f"EXP-78 -- {len(rows)} (target, draw) cells over {len(set(r[0] for r in rows))} weak fleet targets")
print(f"  SUBSET probe  (3-sets at the committed dose): {gS.mean():+6.2f}pp (SE {se(gS):.2f})  improving {int((gS>0).sum())}/{len(gS)}  abstained {sum(r[5] for r in rows)}")
print(f"  SINGLETON probe (the deployed method)       : {gP.mean():+6.2f}pp (SE {se(gP):.2f})  improving {int((gP>0).sum())}/{len(gP)}  abstained {sum(r[6] for r in rows)}")
print(f"  PAIRED subset - singleton                   : {d.mean():+6.2f}pp (SE {se(d):.2f}, p={stats.ttest_1samp(d,0).pvalue:.3f})  subset better {int((d>0).sum())} / worse {int((d<0).sum())} / tie {int((d==0).sum())}")
for k in DRAWS:
    m = [i for i, r in enumerate(rows) if r[1] == k]
    print(f"    draw {k}: n={len(m):2d}  subset {gS[m].mean():+6.2f}  singleton {gP[m].mean():+6.2f}  paired {d[m].mean():+6.2f}")
# per-target draw-averaged
T = sorted(set(r[0] for r in rows))
dt = np.array([np.mean([d[i] for i, r in enumerate(rows) if r[0] == t]) for t in T])
print(f"  per-target draw-averaged paired diff (n={len(T)}): {dt.mean():+6.2f}pp (SE {se(dt):.2f}, p={stats.ttest_1samp(dt,0).pvalue:.3f})")
