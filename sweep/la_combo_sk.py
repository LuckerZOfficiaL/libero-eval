#!/usr/bin/env python3
"""EXP-97 with scikit-learn: ridge (alpha by grouped CV) + gradient boosting (nonlinear upper bound),
leave-one-receiver-out, same features/labels as la_combo.py."""
import pathlib, sys, numpy as np
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import la_combo as C
from la_combo import A, R, WEAK, U_
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def evaluate(F, label, L):
    T = [t for t in U_ if not np.isnan(np.asarray(F[sorted(F)[0]])[t, [d for d in U_ if d != t]]).all()]
    X, Y, G, names = C.build(F, T)
    models = {"ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13))),
              "gboost": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=0)}
    for mname, mk in models.items():
        rho = {}; p1, p3, rnd, best = [], [], [], []
        for t in T:
            tr, te = G != t, G == t
            p = mk().fit(X[tr], Y[tr]).predict(X[te])
            ds = [d for d in U_ if d != t]; y = np.array([A.rec(t, d)["success_rate"] for d in ds])
            rho[t] = stats.spearmanr(p, y).statistic if y.std() > 0 else np.nan
            if t in WEAK:
                b, _ = R.base(t); o = np.argsort(-p)
                p1.append(100 * (y[o[0]] - b)); p3.append(100 * (y[o[:3]].mean() - b)); rnd.append(100 * (y.mean() - b)); best.append(100 * (y.max() - b))
        rw = np.array([rho[t] for t in T if t in WEAK]); ra = np.array([v for v in rho.values() if not np.isnan(v)])
        L.append(f"  {label:<16} {mname:<7} weak rho {rw.mean():+.3f} (SE {rw.std(ddof=1)/np.sqrt(len(rw)):.3f}, p={stats.ttest_1samp(rw,0).pvalue:.2f}, {int((rw>0).sum())}/{len(rw)} up)"
                 f" | all rho {ra.mean():+.3f} (p={stats.ttest_1samp(ra,0).pvalue:.3f})"
                 f" | weak picks: top-1 {np.mean(p1):+.1f}pp, top-3 {np.mean(p3):+.1f}pp, random {np.mean(rnd):+.1f}, best {np.mean(best):+.1f}")
        if mname == "ridge":
            m = mk().fit(X, Y); co = m[-1].coef_; top = np.argsort(-np.abs(co))[:6]
            L.append(f"      ridge alpha {m[-1].alpha_:.3g}; largest weights: " + ", ".join(f"{names[i]} {co[i]:+.3f}" for i in top))

L = ["=== EXP-97 (scikit-learn): combinations of metrics vs ground-truth donor ranking, leave-one-receiver-out ===",
     "ceilings: weak 0.795, all 0.519. Best single metric: weak |rho| 0.11, all 0.08"]
la = C.la_feats(); en = C.energy_feats()
evaluate(la, "LA (72)", L); evaluate({**la, **en}, "LA + energy (75)", L)
txt = "\n".join(L); print(txt); (C.REPO / "results/analysis/exp97_la_combo_sk_report.txt").write_text(txt + "\n")
