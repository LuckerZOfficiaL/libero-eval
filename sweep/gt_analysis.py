#!/usr/bin/env python3
"""EXP-95 analysis: the ground-truth donor matrix (e98_t{t}_d{d}, every donor tried once per receiver).

1. how reliable is the ranking?   split the 25 initial states into halves, rank donors on each half,
   Spearman between halves (random splits), Spearman-Brown to the full 25 -> the ceiling for any predictor
2. what does the landscape look like?   spread, share of donors that help / hurt, best donor gain
3. do the old noisy probe labels agree with it?   (1,537 labels, 10-rollout probes)
4. what linear-algebra quantity correlates with it?   every EXP-91 weight score and EXP-94 data-aware score,
   within-target Spearman over all 73 donors, all targets (no selection, no split: every score reported)
"""
import json, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from reinforce_train import scenes as SC
import rank_sweep as R
import la_pair_eval as E
UNIQUE = [t for t in range(76) if t not in (18, 62)]

def rec(t, d):
    f = REPO / f"results/merge/e98_t{t}_d{d}.jsonl"
    if not f.exists(): return None
    for ln in open(f):
        r = json.loads(ln)
        if r.get("status") == "ok" and r["scene"] == SC[t][0]: return r
    return None

def main(min_donors=73, subset=None, tag=""):
    G = {}
    for t in (subset or UNIQUE):
        rs = {d: rec(t, d) for d in UNIQUE if d != t}
        rs = {d: r for d, r in rs.items() if r is not None}
        if len(rs) >= min_donors: G[t] = rs
    T = sorted(G); L = []; P = lambda *a: L.append(" ".join(str(x) for x in a))
    P(f"=== EXP-95 ground truth{tag}: {len(T)} receivers with all {min_donors} donors ===")
    # 1. reliability
    rng = np.random.default_rng(95); half, full = [], []
    for t in T:
        ds = sorted(G[t]); S = np.array([[1.0 if s >= 0 else 0.0 for s in G[t][d]["success_step"]] for d in ds])
        n = S.shape[1]; rr = []
        for _ in range(200):
            p = rng.permutation(n); a, b = S[:, p[:n // 2]].mean(1), S[:, p[n // 2:]].mean(1)
            if a.std() > 0 and b.std() > 0: rr.append(stats.spearmanr(a, b).statistic)
        rh = float(np.mean(rr)) if rr else np.nan; half.append(rh); full.append(2 * rh / (1 + rh))
    half, full = np.array(half), np.array(full)
    P(f"\n1. reliability of the donor ranking (split-half over the {n} states, Spearman-Brown to {n}):")
    P(f"   half vs half rho {np.nanmean(half):.3f}; full-length reliability {np.nanmean(full):.3f} (range {np.nanmin(full):.2f}-{np.nanmax(full):.2f})")
    P(f"   ceiling for any predictor = sqrt(reliability) = {np.sqrt(np.nanmean(full)):.3f}")
    # 2. landscape
    P("\n2. landscape per receiver (25 rollouts per donor):")
    P(f"   {'t':>4} {'base':>5} {'mean':>5} {'min':>5} {'max':>5} {'help>+8':>8} {'hurt<-8':>8}")
    for t in T:
        b, _ = R.base(t); v = np.array([G[t][d]["success_rate"] for d in sorted(G[t])])
        P(f"   {t:>4} {100*b:>4.0f}% {100*v.mean():>4.0f}% {100*v.min():>4.0f}% {100*v.max():>4.0f}% {int((v-b>0.08).sum()):>8} {int((v-b<-0.08).sum()):>8}")
    # 3. old labels vs ground truth
    lab = E.labels(); ro = []
    for t in T:
        ds = [d for d in sorted(G[t]) if (t, d) in lab]
        if len(ds) >= 5:
            x = [lab[(t, d)] for d in ds]; y = [G[t][d]["success_rate"] for d in ds]
            if np.std(x) > 0 and np.std(y) > 0: ro.append(stats.spearmanr(x, y).statistic)
    P(f"\n3. old probe labels vs ground truth, within target: rho {np.mean(ro):+.3f} (n={len(ro)} receivers)")
    # 4. linear-algebra scores
    z = np.load(REPO / "results/analysis/la_pair_features.npz")
    def agg(k): x = z[k]; return np.concatenate([x.sum(0, keepdims=True), x], 0)
    ip, aip, rn2, dn2 = agg("ip"), agg("absip"), agg("rn2"), agg("dn2")
    feats = {}
    for bnd in range(6):
        rt, rd = rn2[bnd][:, None], rn2[bnd][None, :]; B = E.BANDS[bnd]
        feats[f"cos_r|{B}"] = ip[bnd] / np.sqrt(rt * rd); feats[f"sign_agree|{B}"] = ip[bnd] / np.maximum(aip[bnd], 1e-30)
        feats[f"cos_D|{B}"] = agg("ipD")[bnd] / np.sqrt(dn2[bnd][:, None] * dn2[bnd][None, :])
        nm = np.maximum(agg("OL")[bnd].diagonal()[:, None], 1e-30)
        feats[f"overlap_out|{B}"] = agg("OL")[bnd] / nm; feats[f"overlap_in|{B}"] = agg("OR")[bnd] / nm
        feats[f"donor_in_target_2s|{B}"] = agg("DinT2")[bnd] / rd; feats[f"donor_in_target_out|{B}"] = agg("DinTL")[bnd] / rd
        feats[f"donor_in_target_in|{B}"] = agg("DinTR")[bnd] / rd; feats[f"target_in_donor_2s|{B}"] = agg("DinT2")[bnd].T / rt
        feats[f"donor_resid_norm|{B}"] = np.broadcast_to(np.sqrt(rd), (76, 76))
    zi = np.load(REPO / "results/analysis/la_input_score.npz")
    mag, dr, rni = [np.concatenate([zi[k].sum(0, keepdims=True), zi[k]], 0) for k in ("mag", "dir", "rn2")]
    for bnd in range(6):
        self_ = np.array([mag[bnd, t, t] for t in range(76)])[:, None]
        feats[f"IN_gain|{E.BANDS[bnd]}"] = mag[bnd] / np.maximum(rni[bnd][None, :], 1e-30)
        feats[f"IN_align|{E.BANDS[bnd]}"] = dr[bnd] / np.sqrt(np.maximum(mag[bnd] * self_, 1e-60))
    res = []
    for nm_, F in feats.items():
        rr = []
        for t in T:
            ds = sorted(G[t]); y = np.array([G[t][d]["success_rate"] for d in ds]); x = F[t, ds]
            if y.std() > 0 and np.std(x) > 0: rr.append(stats.spearmanr(x, y).statistic)
        rr = np.array(rr); res.append((nm_, rr.mean(), rr.std(ddof=1) / np.sqrt(len(rr)), (rr > 0).sum(), len(rr)))
    res.sort(key=lambda r: -abs(r[1]))
    P(f"\n4. linear-algebra scores vs ground truth, within-target Spearman over all {min_donors} donors ({len(res)} scores, all reported):")
    for nm_, m, se, up, n_ in res[:20]:
        P(f"   {nm_:<30} rho {m:+.3f} (SE {se:.3f}, positive on {up}/{n_})")
    P(f"   ... median |rho| {np.median([abs(r[1]) for r in res]):.3f}; ceiling {np.sqrt(np.nanmean(full)):.3f}")
    txt = "\n".join(L); print(txt); (REPO / f"results/analysis/exp95_ground_truth_report{tag.replace(' ', '_')}.txt").write_text(txt + "\n")

if __name__ == "__main__":
    from subset_probe import TARGETS as WEAK
    main(subset=WEAK, tag=" weak (25 rollouts)")
    main(subset=[t for t in UNIQUE if t not in WEAK], tag=" strong (10 rollouts)")
    main(tag=" all")
