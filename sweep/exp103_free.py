#!/usr/bin/env python3
"""EXP-103b: zero-cost checks suggested by the four mergeability papers. CPU, no rollouts, no new features
except the gradient dot product (from EXP-99's saved sketches).

1. donor-only ceiling: a donor's mean effect on the OTHER receivers, as a ranking for a new receiver
   (covers every donor-only property: Rahamim et al.'s base-knowledge quantities, sigma_max, donor norm ...)
2. effect size (Model Kinship, Hu et al.): every existing score vs |success(t+d) - success(t alone)|
3. encoder-gradient DOT product (Demystifying Mergeability #25), never scored
4. correlation-objective L1 combination (Demystifying Mergeability's predictor): linear score fit to maximise
   the mean WITHIN-RECEIVER Pearson, L1 penalty, leave-one-receiver-out; trained on all 74 receivers
   (categories A,B,C,E,F + spectral EXP-103a if present), tested on the 15 weak ones.
Report results/analysis/exp103_free_report.txt
"""
import pathlib, sys
import numpy as np, torch
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import all_scores_gt as AS
import gt_analysis as A_
import rank_sweep as R
import proxy_eval as PE
from subset_probe import TARGETS as WEAK
UN = A_.UNIQUE; WK = [t for t in UN if t in WEAK]
GT = {t: {d: A_.rec(t, d)["success_rate"] for d in UN if d != t} for t in UN}
L = []; P = lambda *a: (L.append(" ".join(str(x) for x in a)), print(*a, flush=True))

def within(M, T, target):
    rr = []
    for t in T:
        ds = sorted(GT[t]); y = np.array([target(t, d) for d in ds]); x = np.asarray(M[t, ds], float)
        if np.isnan(x).all() or y.std() == 0 or np.nanstd(x) == 0: continue
        rr.append(stats.spearmanr(x, y, nan_policy="omit").statistic)
    return np.array(rr)

def features(with_spec=True):
    F = {}
    for cname, f in (("A", AS.cat_A), ("B", AS.cat_B), ("C", AS.cat_C), ("D", AS.cat_D), ("E", AS.cat_E), ("F", AS.cat_F)):
        for k, v in f().items(): F[f"{cname}:{k}"] = np.asarray(v, float)[:76, :76]
    sp = REPO / "results/analysis/exp103_spec.npz"
    if with_spec and sp.exists():
        z = np.load(sp)
        for k in ("specov", "botR", "botI", "topR", "botL"):
            for b in range(6): F[f"S:{k}|{AS.BANDS[b]}"] = z[k][b]
        for b in range(6): F[f"S:smax_d|{AS.BANDS[b]}"] = np.broadcast_to(z["smax"][b][None, :], (76, 76)).copy()
    return F

def main():
    base = {t: R.base(t)[0] for t in UN}
    gain = lambda t, d: GT[t][d] - base[t]
    P("=== EXP-103b zero-cost checks (within-receiver Spearman; weak = 15 receivers, 25-rollout GT, ceiling 0.80) ===")
    # 1. donor-only ceiling
    P("\n1. donor-only ceiling: donor's mean centred effect on the OTHER receivers")
    for nm, Tp in (("from all other receivers", UN), ("from other weak receivers", WK)):
        M = np.full((76, 76), np.nan)
        for t in UN:
            for d in GT[t]:
                M[t, d] = np.mean([GT[u][d] - np.mean(list(GT[u].values())) for u in Tp if u != t and d in GT[u]])
        r = within(M, WK, lambda t, d: GT[t][d]); ra = within(M, UN, lambda t, d: GT[t][d])
        P(f"   {nm:28s} weak rho {r.mean():+.3f} (p={stats.ttest_1samp(r,0).pvalue:.3f})   all 74 rho {ra.mean():+.3f} (p={stats.ttest_1samp(ra,0).pvalue:.3f})")
    F = features()
    P("\n0. NEW spectral scores from Demystifying Mergeability (EXP-103a) vs signed success")
    for k in [k for k in F if k.startswith("S:")]:
        r = within(F[k], WK, lambda t, d: GT[t][d]); ra = within(F[k], UN, lambda t, d: GT[t][d])
        P(f"   {k:22s} weak {r.mean():+.3f} (p={stats.ttest_1samp(r,0).pvalue:.3f})   all {ra.mean():+.3f} (p={stats.ttest_1samp(ra,0).pvalue:.3f})")
    # 2. effect size
    P(f"\n2. effect size |gain| (Model Kinship): {len(F)} scores vs |success(t+d) - success(t alone)|, weak receivers")
    res = []
    for k, M in F.items():
        r_abs = within(M, WK, lambda t, d: abs(gain(t, d))); r_sig = within(M, WK, lambda t, d: GT[t][d])
        if len(r_abs): res.append((k, r_abs.mean(), r_sig.mean() if len(r_sig) else np.nan, len(r_abs)))
    res.sort(key=lambda x: -abs(x[1]))
    P(f"   best |rho| vs |gain| {abs(res[0][1]):.3f}; median {np.median([abs(x[1]) for x in res]):.3f}; >= 0.15: {sum(abs(x[1]) >= 0.15 for x in res)}")
    for k, ra, rs, n in res[:10]: P(f"   {k:34s} vs |gain| {ra:+.3f}   vs signed success {rs:+.3f}   (n={n})")
    # kinship sign check: similarity vs |gain| on all receivers (10-rollout noise there)
    for k in ("A:cos_r|all", "A:cos_D|all"):
        if k in F:
            ra = within(F[k], UN, lambda t, d: abs(gain(t, d))); P(f"   {k} vs |gain|, all 74: {ra.mean():+.3f} (p={stats.ttest_1samp(ra,0).pvalue:.3f})")
    # 3. gradient dot
    P("\n3. encoder-gradient dot product (sketches from EXP-99)")
    S = PE.merge("grad_proxy_s*.npz", ["sketch"])["sketch"]
    zb = np.load(sorted((REPO / "results/analysis").glob("grad_proxy_s*.npz"))[0])["band_of"]
    skb = S.reshape(76, len(zb), -1)
    for b in range(6):
        m = np.ones(len(zb), bool) if b == 0 else (zb == b - 1)
        X = skb[:, m, :].reshape(76, -1); D = X @ X.T
        r = within(D, WK, lambda t, d: GT[t][d]); ra = within(D, UN, lambda t, d: GT[t][d])
        P(f"   grad_dot|{AS.BANDS[b]:9s} weak {r.mean():+.3f}   all {ra.mean():+.3f}")
    # 4. correlation-objective L1 combination, leave-one-receiver-out
    P("\n4. correlation-objective L1 combination (Demystifying Mergeability), leave-one-receiver-out")
    keys = [k for k, M in F.items() if not k.startswith("D:") and np.isfinite(M[np.ix_(UN, UN)][~np.eye(len(UN), dtype=bool)]).mean() > 0.95]
    def zrank(t, k):
        ds = sorted(GT[t]); x = np.asarray(F[k][t, ds], float)
        if not np.isfinite(x).any(): return np.zeros(len(ds))
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        x = stats.rankdata(x); return (x - x.mean()) / (x.std() + 1e-9)
    Xt = {t: torch.tensor(np.stack([zrank(t, k) for k in keys], 1), dtype=torch.float32) for t in UN}
    Yt = {t: torch.tensor(stats.rankdata([GT[t][d] for d in sorted(GT[t])]), dtype=torch.float32) for t in UN}
    def fit(T, lam, ep=300):
        w = torch.zeros(len(keys), requires_grad=True); opt = torch.optim.Adam([w], lr=0.01)
        with torch.no_grad(): w.normal_(0, 0.01)
        for _ in range(ep):
            loss = 0
            for t in T:
                s = Xt[t] @ w; y = Yt[t]
                s = s - s.mean(); yc = y - y.mean()
                loss = loss - (s * yc).sum() / (s.norm() * yc.norm() + 1e-9)
            loss = loss / len(T) + lam * w.abs().sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return w.detach()
    for lam in (1e-3, 1e-2):
        out = []
        for t in WK:
            w = fit([u for u in UN if u != t], lam)
            s = (Xt[t] @ w).numpy(); out.append(stats.spearmanr(s, Yt[t].numpy()).statistic)
        out = np.array(out)
        P(f"   lambda {lam:g}: {len(keys)} features, held-out weak rho {out.mean():+.3f} (p={stats.ttest_1samp(out,0).pvalue:.3f}), "
          f"positive on {int((out>0).sum())}/{len(out)}")
    (REPO / "results/analysis/exp103_free_report.txt").write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
