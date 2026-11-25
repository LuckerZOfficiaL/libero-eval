#!/usr/bin/env python3
"""EXP-97 — can a COMBINATION of the linear-algebra (and energy) metrics predict donor goodness?

Features per (receiver t, donor d): the 72 LA scores of EXP-91/94 (gt_analysis) + the 4 EXP-96 energy signals,
each z-scored WITHIN receiver (only the ranking of donors for a receiver matters). Label: ground-truth success
of t with donor d (EXP-95), z-scored within receiver. Models: ridge (alpha chosen by inner CV), and gradient
boosting as a nonlinear upper bound on what these features contain. Leave-one-receiver-out: every prediction
is for a receiver the model never saw. Score: within-receiver Spearman vs ground truth (weak receivers have the
reliable 25-rollout GT; ceiling 0.795), and the value of picking the top-1 / top-3 predicted donors.
"""
import glob, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import gt_analysis as A, rank_sweep as R, la_pair_eval as E
from subset_probe import TARGETS as WEAK
import torch

class RidgeCV:
    """numpy ridge with the penalty chosen by 5-fold CV (folds = receivers) inside the training set."""
    def __init__(self, alphas, groups=None): self.alphas, self.groups = alphas, groups
    def _fit(self, X, Y, a):
        mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
        w = np.linalg.solve(Xs.T @ Xs + a * np.eye(X.shape[1]), Xs.T @ (Y - Y.mean()))
        return mu, sd, w, Y.mean()
    def fit(self, X, Y, groups=None):
        g = groups if groups is not None else np.arange(len(Y)) % 5
        ug = np.unique(g); folds = [ug[i::5] for i in range(5)]
        best = None
        for a in self.alphas:
            err = 0.0
            for f in folds:
                te = np.isin(g, f); mu, sd, w, b = self._fit(X[~te], Y[~te], a)
                err += np.sum((((X[te] - mu) / sd) @ w + b - Y[te]) ** 2)
            if best is None or err < best[0]: best = (err, a)
        self.alpha_ = best[1]; self.mu, self.sd, self.coef_, self.b = self._fit(X, Y, self.alpha_); return self
    def predict(self, X): return ((X - self.mu) / self.sd) @ self.coef_ + self.b

class MLP:
    """nonlinear upper bound: 2-layer MLP, early-stopped on a held-out 20% of training receivers."""
    def fit(self, X, Y, groups=None):
        torch.manual_seed(0); g = groups; ug = np.unique(g); rng = np.random.default_rng(0)
        va = np.isin(g, rng.choice(ug, max(1, len(ug) // 5), replace=False))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-2); best = (1e9, None)
        tr = torch.tensor(~va); vv = torch.tensor(va)
        for ep in range(200):
            self.net.train(); perm = torch.randperm(int(tr.sum()))
            Xa, Ya = Xt[tr][perm], Yt[tr][perm]
            for i in range(0, len(Ya), 256):
                opt.zero_grad(); loss = ((self.net(Xa[i:i+256]).squeeze(1) - Ya[i:i+256]) ** 2).mean(); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad(): v = float(((self.net(Xt[vv]).squeeze(1) - Yt[vv]) ** 2).mean())
            if v < best[0]: best = (v, {k: x.clone() for k, x in self.net.state_dict().items()})
        self.net.load_state_dict(best[1]); return self
    def predict(self, X):
        with torch.no_grad(): return self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)).squeeze(1).numpy()
U_ = A.UNIQUE

def la_feats():
    z = np.load(REPO / "results/analysis/la_pair_features.npz")
    def agg(k): x = z[k]; return np.concatenate([x.sum(0, keepdims=True), x], 0)
    ip, aip, rn2, dn2 = agg("ip"), agg("absip"), agg("rn2"), agg("dn2"); F = {}
    for b in range(6):
        rt, rd = rn2[b][:, None], rn2[b][None, :]; B = E.BANDS[b]
        F[f"cos_r|{B}"] = ip[b] / np.sqrt(rt * rd); F[f"sign_agree|{B}"] = ip[b] / np.maximum(aip[b], 1e-30)
        F[f"cos_D|{B}"] = agg("ipD")[b] / np.sqrt(dn2[b][:, None] * dn2[b][None, :])
        nm = np.maximum(agg("OL")[b].diagonal()[:, None], 1e-30)
        F[f"overlap_out|{B}"] = agg("OL")[b] / nm; F[f"overlap_in|{B}"] = agg("OR")[b] / nm
        F[f"dinT2|{B}"] = agg("DinT2")[b] / rd; F[f"dinTL|{B}"] = agg("DinTL")[b] / rd; F[f"dinTR|{B}"] = agg("DinTR")[b] / rd
        F[f"tinD2|{B}"] = agg("DinT2")[b].T / rt; F[f"dnorm|{B}"] = np.broadcast_to(np.sqrt(rd), (76, 76))
    zi = np.load(REPO / "results/analysis/la_input_score.npz")
    mag, dr, rni = [np.concatenate([zi[k].sum(0, keepdims=True), zi[k]], 0) for k in ("mag", "dir", "rn2")]
    for b in range(6):
        self_ = np.array([mag[b, t, t] for t in range(76)])[:, None]
        F[f"IN_gain|{E.BANDS[b]}"] = mag[b] / np.maximum(rni[b][None, :], 1e-30)
        F[f"IN_align|{E.BANDS[b]}"] = dr[b] / np.sqrt(np.maximum(mag[b] * self_, 1e-60))
    return F

def energy_feats():
    F = {}
    for f in sorted(glob.glob(str(REPO / "results/analysis/energy_proxy_s*.npz"))):
        z = np.load(f)
        for s in ("disp", "drift", "demo"):
            F[s] = z[s][:, :76].copy() if s not in F else np.where(np.isnan(F[s]), z[s][:, :76], F[s])
    return F

def zs(x):
    sd = np.nanstd(x); return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)

def build(F, T):
    names = sorted(F); X, Y, G = [], [], []
    for t in T:
        ds = [d for d in U_ if d != t]
        y = np.array([A.rec(t, d)["success_rate"] for d in ds])
        cols = [zs(np.asarray(F[n][t, ds], float)) for n in names]
        X.append(np.nan_to_num(np.stack(cols, 1))); Y.append(zs(y)); G += [t] * len(ds)
    return np.vstack(X), np.concatenate(Y), np.array(G), names

def evaluate(F, label, L):
    T = [t for t in U_ if not np.isnan(np.asarray(F[sorted(F)[0]])[t, [d for d in U_ if d != t]]).all()]
    X, Y, G, names = build(F, T)
    res = {}
    for mname, mk in (("ridge", lambda: RidgeCV(alphas=np.logspace(-1, 5, 13))), ("mlp", lambda: MLP())):
        rho = {}; picks1, picks3, rnd, best = [], [], [], []
        for t in T:
            tr, te = G != t, G == t
            m = mk().fit(X[tr], Y[tr], G[tr]); p = m.predict(X[te])
            ds = [d for d in U_ if d != t]; y = np.array([A.rec(t, d)["success_rate"] for d in ds])
            rho[t] = stats.spearmanr(p, y).statistic if y.std() > 0 else np.nan
            if t in WEAK:
                b, _ = R.base(t); o = np.argsort(-p)
                picks1.append(100 * (y[o[0]] - b)); picks3.append(100 * (y[o[:3]].mean() - b)); rnd.append(100 * (y.mean() - b)); best.append(100 * (y.max() - b))
        rw = np.array([rho[t] for t in T if t in WEAK]); ra = np.array([v for v in rho.values() if not np.isnan(v)])
        L.append(f"  {label:<14} {mname:<9} weak rho {rw.mean():+.3f} (SE {rw.std(ddof=1)/np.sqrt(len(rw)):.3f}, p={stats.ttest_1samp(rw,0).pvalue:.2f}, {int((rw>0).sum())}/{len(rw)} up)"
                 f" | all rho {ra.mean():+.3f} (p={stats.ttest_1samp(ra,0).pvalue:.3f})"
                 f" | weak picks: top-1 {np.mean(picks1):+.1f}pp, top-3 {np.mean(picks3):+.1f}pp, random {np.mean(rnd):+.1f}, best {np.mean(best):+.1f}")
        if mname == "ridge":
            m = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(X, Y, G)
            top = np.argsort(-np.abs(m.coef_))[:6]
            L.append(f"      ridge alpha {m.alpha_:.3g}; largest weights: " + ", ".join(f"{names[i]} {m.coef_[i]:+.3f}" for i in top))
    return res

def main():
    L = ["=== EXP-97: linear combinations of the metrics vs the ground-truth donor ranking (leave-one-receiver-out) ===",
         "ceilings: weak 0.795, all 0.519. Single best metric: weak |rho| 0.11, all 0.08"]
    la = la_feats(); en = energy_feats()
    evaluate(la, "LA (72)", L)
    evaluate({**la, **en}, "LA + energy (75)", L)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp97_la_combo_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
