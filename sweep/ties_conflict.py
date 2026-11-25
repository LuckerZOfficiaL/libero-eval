#!/usr/bin/env python3
"""EXP-100 — TIES-style elementwise sign conflicts between target and donor residuals (weights only, CPU).

TIES-merging attributes interference to parameters where task vectors disagree in SIGN among their large
entries. Our earlier 'sign agreement' was at module level (sign of <r_t, r_d> per module). Here, element level:
per module, P random parameter positions (i, j), residuals r = D - P16(D) evaluated there from the factors;
for every pair (t, d) and band:
  conflict  = among positions where BOTH are in their own top-20% |entries| (TIES 'trim'), share with opposite sign
  overlap   = share of t's top-20% positions that are also in d's top-20% (support overlap)
  agree_all = share of ALL sampled positions with the same sign, |r|-weighted
Counts are weighted by module size / P so every parameter counts once in expectation.
Output results/analysis/ties_conflict.npz; evaluation against the EXP-95 ground truth at the end.
"""
import json, pathlib, sys, time
import numpy as np, torch
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import la_pair_features as LF
from reinforce_train import group_of
torch.set_num_threads(48)
N, NB, P, KEEP = 76, 5, 20000, 0.2

def main():
    t0 = time.time()
    AB = [LF.load(t) for t in range(N)]
    stems = sorted(set.intersection(*[set(x) for x in AB]))
    acc = {k: np.zeros((NB, N, N)) for k in ("conf_n", "conf_d", "ovl_n", "ovl_d", "agr_n", "agr_d")}
    rng = np.random.default_rng(100)
    for mi, stem in enumerate(stems):
        g = group_of(stem)
        b = torch.stack([AB[t][stem][0] for t in range(N)]).double(); a = torch.stack([AB[t][stem][1] for t in range(N)]).double()
        dout, din = b.shape[1], a.shape[2]
        Ball = b.permute(1, 0, 2).reshape(dout, N * 16); Aall = a.reshape(N * 16, din)
        Qb, Rb = torch.linalg.qr(Ball); Qa, Ra = torch.linalg.qr(Aall.T); Us, sv, Vh = torch.linalg.svd(Rb @ Ra.T)
        U16 = Qb @ Us[:, :16]; V16 = Qa @ Vh[:16].T
        Ub = U16.T @ b; aV = a @ V16
        Bp = torch.cat([b, -(U16 @ Ub)], 2); Ap = torch.cat([a, aV @ V16.T], 1)       # r = Bp @ Ap
        ii = torch.from_numpy(rng.integers(0, dout, P)); jj = torch.from_numpy(rng.integers(0, din, P))
        R = (Bp[:, ii, :] * Ap[:, :, jj].transpose(1, 2)).sum(-1)                        # [N, P] residual entries
        w = dout * din / P
        A_ = R.abs(); thr = torch.quantile(A_, 1 - KEEP, dim=1, keepdim=True)
        keep = (A_ >= thr).double(); sg = torch.sign(R)
        both = keep @ keep.T                                                               # co-kept counts [N,N]
        same_kept = (keep * sg) @ (keep * sg).T                                            # (#same - #opp) among co-kept
        acc["conf_n"][g] += w * ((both - same_kept) / 2).numpy(); acc["conf_d"][g] += w * both.numpy()
        acc["ovl_n"][g] += w * both.numpy(); acc["ovl_d"][g] += w * keep.sum(1, keepdim=True).expand(N, N).numpy()
        wa = (A_[:, None, :] * A_[None, :, :]).sqrt()                                      # |r_t r_d|^1/2 weights [N,N,P]
        acc["agr_n"][g] += w * ((sg[:, None, :] * sg[None, :, :] > 0).double() * wa).sum(-1).numpy()
        acc["agr_d"][g] += w * wa.sum(-1).numpy()
        if mi % 50 == 0: print(f"[tc] {mi+1}/{len(stems)} {(time.time()-t0)/60:.1f} min", flush=True)
    np.savez(REPO / "results/analysis/ties_conflict.npz", **acc)
    print(f"[tc] DONE {len(stems)} modules in {(time.time()-t0)/60:.1f} min", flush=True)
    evaluate()

def evaluate():
    import gt_analysis as A
    from subset_probe import TARGETS as WEAK
    z = np.load(REPO / "results/analysis/ties_conflict.npz")
    def ag(k): x = z[k]; return np.concatenate([x.sum(0, keepdims=True), x], 0)
    bands = ["all", "vision", "llm_early", "llm_late", "expert", "head"]
    F = {}
    for b in range(6):
        F[f"conflict|{bands[b]}"] = ag("conf_n")[b] / np.maximum(ag("conf_d")[b], 1e-9)
        F[f"overlap|{bands[b]}"] = ag("ovl_n")[b] / np.maximum(ag("ovl_d")[b], 1e-9)
        F[f"agree_all|{bands[b]}"] = ag("agr_n")[b] / np.maximum(ag("agr_d")[b], 1e-9)
    GT = {t: {d: A.rec(t, d)["success_rate"] for d in A.UNIQUE if d != t} for t in A.UNIQUE}
    L = ["=== EXP-100: TIES elementwise sign conflict / support overlap vs EXP-95 ground truth (within-receiver Spearman) ===",
         "ceilings: weak 0.795, all 0.519"]
    res = []
    for nm, M_ in F.items():
        out = {}
        for grp, T in (("weak", [t for t in A.UNIQUE if t in WEAK]), ("all", A.UNIQUE)):
            rr = []
            for t in T:
                ds = sorted(GT[t]); y = np.array([GT[t][d] for d in ds]); x = M_[t, ds]
                if y.std() > 0 and np.std(x) > 0: rr.append(stats.spearmanr(x, y).statistic)
            out[grp] = np.array(rr)
        res.append((nm, out))
    res.sort(key=lambda r: -abs(r[1]["weak"].mean()))
    for nm, out in res:
        L.append(f"  {nm:<22} weak rho {out['weak'].mean():+.3f} (p={stats.ttest_1samp(out['weak'],0).pvalue:.2f})   all rho {out['all'].mean():+.3f} (p={stats.ttest_1samp(out['all'],0).pvalue:.3f})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp100_ties_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main() if "--eval" not in sys.argv else evaluate()
