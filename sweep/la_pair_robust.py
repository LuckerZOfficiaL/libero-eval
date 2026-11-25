#!/usr/bin/env python3
"""EXP-91 robustness: is the anti-similarity signal just twin conflict? and is it usable as a shortlist?"""
import sys, pathlib, numpy as np
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import la_pair_eval as E
z = np.load(E.REPO / "results/analysis/la_pair_features.npz")
ip = z["ip"]; rn2 = z["rn2"]
def cosb(bands): i = ip[bands].sum(0); r = rn2[bands].sum(0); return i / np.sqrt(r[:, None] * r[None, :])
F = {"cos_r|llm_early": cosb([1]), "cos_r|all": cosb([0, 1, 2, 3, 4])}
lab = E.labels(); by = {}
for (t, d), v in lab.items(): by.setdefault(t, {})[d] = v
T = [t for t in sorted(by) if len(by[t]) >= 8 and t not in (18, 62)]
out = []
for nm, C in F.items():
    for drop in (0, 1, 3):
        rh = []
        for t in T:
            ds = sorted(by[t], key=lambda d: -E.np.nan_to_num(C[t, d]))[drop:]      # drop the `drop` most-similar donors
            y = np.array([by[t][d] for d in ds]); x = -C[t, ds]
            if y.std() > 0: rh.append(stats.spearmanr(x, y).statistic)
        rh = np.array(rh); out.append(f"{nm:<18} drop {drop} most-similar: rho(-cos, label) {rh.mean():+.3f} (SE {rh.std(ddof=1)/np.sqrt(len(rh)):.3f}, p={stats.ttest_1samp(rh,0).pvalue:.3f}, n={len(rh)})")
# shortlist: random 8-candidate menus; keep the 4 LEAST similar; is the label-best donor kept more often than chance (50%)?
rng = np.random.default_rng(5); C = F["cos_r|llm_early"]; hold, val_s, val_r = [], [], []
for t in T:
    ds = sorted(by[t])
    for _ in range(50):
        m = rng.choice(ds, 8, replace=False); y = np.array([by[t][d] for d in m])
        keep = m[np.argsort(C[t, m])[:4]]; rkeep = rng.choice(m, 4, replace=False)
        best = m[np.argmax(y)]; hold.append(best in keep)
        val_s.append(np.sort([by[t][d] for d in keep])[::-1][:3].mean()); val_r.append(np.sort([by[t][d] for d in rkeep])[::-1][:3].mean())
out.append(f"shortlist 8->4 by LEAST llm_early similarity: keeps the label-best donor {100*np.mean(hold):.1f}% (chance 50%); "
           f"best-3 within shortlist {100*np.mean(val_s):.2f}% vs random-4 {100*np.mean(val_r):.2f}% ({100*(np.mean(val_s)-np.mean(val_r)):+.2f}pp)")
txt = "\n".join(out); print(txt); (E.REPO / "results/analysis/exp91_robustness.txt").write_text(txt + "\n")
