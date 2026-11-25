#!/usr/bin/env python3
"""Evaluate donor-compatibility proxies (EXP-98 sensitivity, EXP-99 gradient/Fisher) against the EXP-95 ground
truth. Pre-registered sign per signal (score = sign * value; higher score = predicted better donor).
Usage: proxy_eval.py sens | grad"""
import glob, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import gt_analysis as A, rank_sweep as R
import unified_pool_driver as U
from subset_probe import TARGETS as WEAK
BANDS = ["all", "vision", "llm_early", "llm_late", "expert", "head"]

def merge(pattern, keys):
    out = {}
    for f in sorted(glob.glob(str(REPO / f"results/analysis/{pattern}"))):
        z = np.load(f)
        for k in keys:
            x = z[k].astype(float)
            out[k] = x.copy() if k not in out else np.where(np.isnan(out[k]), x, out[k])
    return out

def sens_scores():
    z = merge("sens_proxy_s*.npz", ["sens_h", "sens_v", "sens_img", "rel_img", "rel_h"])
    return {k: (-1, v[:, :76]) for k, v in z.items()}                       # more sensitive = worse

def grad_scores():
    z = merge("grad_proxy_s*.npz", ["proj", "sketch"])
    zb = np.load(sorted(glob.glob(str(REPO / "results/analysis/grad_proxy_s*.npz")))[0])["band_of"]
    proj = z["proj"]                                                          # [t, i, band, d]
    c = np.array([13.351 / U.resid_norm(d) for d in range(76)])              # operator dose c_d (ref fixed above)
    P = np.concatenate([np.nansum(proj, 2, keepdims=True), proj], 2)         # band 0 = all
    out = {}
    for b in range(6):
        look = c[None, :] * np.nanmean(P[:, :, b, :], 1); fis = (c[None, :] ** 2) * np.nanmean(P[:, :, b, :] ** 2, 1)
        out[f"lookahead|{BANDS[b]}"] = (-1, look); out[f"fisher|{BANDS[b]}"] = (-1, fis)
    sk = z["sketch"]                                                          # [t, modules*64]
    nmod = len(zb); skb = sk.reshape(76, nmod, -1)
    for b in range(6):
        m = np.ones(nmod, bool) if b == 0 else (zb == b - 1)
        X = skb[:, m, :].reshape(76, -1)
        nrm = np.linalg.norm(X, axis=1); cos = (X @ X.T) / np.outer(nrm, nrm)
        l2 = np.sqrt(np.maximum((nrm[:, None] ** 2) + (nrm[None, :] ** 2) - 2 * X @ X.T, 0))
        out[f"grad_cos|{BANDS[b]}"] = (+1, cos); out[f"grad_l2|{BANDS[b]}"] = (-1, l2)
    return out

def main(which):
    S = sens_scores() if which == "sens" else grad_scores()
    GT = {t: {d: A.rec(t, d)["success_rate"] for d in A.UNIQUE if d != t} for t in A.UNIQUE}
    first = next(iter(S.values()))[1]
    done = [t for t in A.UNIQUE if not np.isnan(first[t, [d for d in A.UNIQUE if d != t]]).any()]
    L = [f"=== {which}: proxy vs EXP-95 ground truth, {len(done)}/74 receivers; score = pre-registered sign x value ===",
         "ceilings: weak 0.795, all 0.519; best previous metric |rho| 0.11"]
    rows = []
    for nm, (sg, M) in S.items():
        res = {}
        for grp, T in (("weak", [t for t in done if t in WEAK]), ("all", done)):
            rr = []
            for t in T:
                ds = sorted(GT[t]); y = np.array([GT[t][d] for d in ds]); x = sg * M[t, ds]
                if y.std() > 0 and np.nanstd(x) > 0: rr.append(stats.spearmanr(x, y, nan_policy="omit").statistic)
            res[grp] = np.array(rr)
        pick = []
        for t in [t for t in done if t in WEAK]:
            ds = sorted(GT[t]); y = np.array([GT[t][d] for d in ds]); x = sg * M[t, ds]; b, _ = R.base(t)
            pick.append(100 * (y[int(np.nanargmax(x))] - b))
        rows.append((nm, res, np.mean(pick) if pick else np.nan))
    rows.sort(key=lambda r: -r[1]["weak"].mean() if len(r[1]["weak"]) else 0)
    for nm, res, pk in rows:
        w, a = res["weak"], res["all"]
        L.append(f"  {nm:<20} weak rho {w.mean():+.3f} (p={stats.ttest_1samp(w,0).pvalue:.2f}, {int((w>0).sum())}/{len(w)} up)"
                 f"   all rho {a.mean():+.3f} (p={stats.ttest_1samp(a,0).pvalue:.3f})   top-1 pick (weak) {pk:+.1f}pp")
    L.append("  reference (weak receivers): random donor -3.8pp, best donor +16.5pp")
    txt = "\n".join(L); print(txt); (REPO / f"results/analysis/exp{'98' if which == 'sens' else '99'}_{which}_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sens")
