#!/usr/bin/env python3
"""ROUTE 1 analysis — does what the MERGE did to the policy predict the probe's verdict?

For each (target, donor): merged signature (results/analysis/merged_sigs/m<t>_<d>.npz) vs the target's
own signature (act_sigs/task<t>.npz, which equals the merge with no donor because S16 + r_target =
D_target). Per hook depth we form three rollout-free features:
    dist  mean over the 180 observations of ||merged - target||
    cos   mean cosine(merged, target)
    dnorm mean relative change in activation norm
Label: the e27 probe affinity (10-rollout success minus baseline, in pp).

The decisive test is WITHIN-TARGET ranking -- ordering one target's own 8 candidates -- because that is
what a selector must do. A pooled correlation could be driven entirely by target-level differences and
would not select anything. Reference points: the probe's own test-retest rho is 0.41-0.59 (EXP-46), and
the best learned selector so far ranks at rho ~ 0.05 on held-out targets.
"""
import json, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from reinforce_train import target_baseline
LAYERS = ["vision", "llm_00", "llm_05", "llm_11", "llm_17", "exp_00", "exp_08", "exp_17", "act_out"]
MS = REPO / "results/analysis/merged_sigs"; AS = REPO / "results/analysis/act_sigs"
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())

def feats(t, d):
    f = MS / f"m{t}_{d}.npz"; g = AS / f"task{t}.npz"
    if not f.exists() or not g.exists(): return None
    M, T = np.load(f), np.load(g)
    out = {}
    for L in LAYERS:
        a = M[L].astype(np.float32); b = T[L].astype(np.float32)
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        da = np.linalg.norm(a - b, axis=1)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        out[f"dist_{L}"] = float(da.mean())
        out[f"cos_{L}"] = float(((a * b).sum(1) / (na * nb + 1e-9)).mean())
        out[f"dnorm_{L}"] = float(((na - nb) / (nb + 1e-9)).mean())
    return out

rows = []
for t in [int(x) for x in RANK if target_baseline(int(x)) < 0.9]:
    b = target_baseline(t)
    for p, s in RANK[str(t)]:
        fe = feats(t, int(p))
        if fe: rows.append((t, int(p), 100 * (s - b), fe))
if not rows: sys.exit("no merged signatures yet")
tgts = sorted({r[0] for r in rows})
names = sorted(rows[0][3])
print(f"=== ROUTE 1: merged-model features vs probe affinity ===")
print(f"{len(rows)} pairs over {len(tgts)} weak targets; {len(names)} features\n")
print(f"{'feature':<16}{'within-target rho':>19}{'(SE)':>8}{'p':>9}{'pooled r':>10}")
best = []
for nm in names:
    rhos = []
    for t in tgts:
        sub = [(r[3][nm], r[2]) for r in rows if r[0] == t]
        if len(sub) >= 4:
            rr = stats.spearmanr([x[0] for x in sub], [x[1] for x in sub]).correlation
            if np.isfinite(rr): rhos.append(rr)
    if len(rhos) < 3: continue
    x = np.array([r[3][nm] for r in rows]); y = np.array([r[2] for r in rows])
    pooled = stats.pearsonr(x, y)[0]
    m, se = float(np.mean(rhos)), float(stats.sem(rhos))
    p = stats.ttest_1samp(rhos, 0).pvalue
    best.append((abs(m), nm, m, se, p, pooled))
    print(f"{nm:<16}{m:>+19.3f}{se:>8.3f}{p:>9.3f}{pooled:>+10.3f}")
best.sort(reverse=True)
print(f"\nstrongest within-target ranker: {best[0][1]} rho={best[0][2]:+.3f} (SE {best[0][3]:.3f}, p={best[0][4]:.3f})")
print(f"reference: probe test-retest rho 0.41-0.59; learned selector on held-out targets ~0.05")
# combined linear model, leave-one-target-out
X = np.array([[r[3][n] for n in names] for r in rows]); Y = np.array([r[2] for r in rows]); G = np.array([r[0] for r in rows])
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
pred = np.zeros(len(Y))
for t in tgts:
    tr, te = G != t, G == t
    w = np.linalg.solve(X[tr].T @ X[tr] + 5.0 * np.eye(X.shape[1]), X[tr].T @ (Y[tr] - Y[tr].mean()))
    pred[te] = X[te] @ w
rh = [stats.spearmanr(pred[G == t], Y[G == t]).correlation for t in tgts]
rh = [r for r in rh if np.isfinite(r)]
print(f"\nridge on all {len(names)} features, leave-one-target-out: within-target rho {np.mean(rh):+.3f} "
      f"(SE {stats.sem(rh):.3f}, p={stats.ttest_1samp(rh,0).pvalue:.3f}, n={len(rh)} targets)")
top3 = [np.mean(sorted(zip(pred[G==t], Y[G==t]), reverse=True)[:3], axis=0)[1] for t in tgts]
rnd = [Y[G==t].mean() for t in tgts]
orc = [np.mean(sorted(Y[G==t], reverse=True)[:3]) for t in tgts]
print(f"pick top-3 by this model: {np.mean(top3):+.2f}pp   random-3 {np.mean(rnd):+.2f}pp   oracle {np.mean(orc):+.2f}pp")
print(f"  -> captures {100*(np.mean(top3)-np.mean(rnd))/(np.mean(orc)-np.mean(rnd)):.0f}% of the oracle gap (probe itself: ~100% by construction here)")
