#!/usr/bin/env python3
"""EXP-104 evaluation: closed-loop-aware behavioural signals vs the EXP-95 ground truth (weak receivers).
Pre-registered direction (score = sign x value, higher = predicted better donor):
  ci -   dci -   corr +   rc -   drc -   cf +   sel +   vinn +   rp_succ -   rp_fail (none, report both)
Within-receiver Spearman of the score vs GT success, and the GT gain (vs target alone) of each score's top-3
donors vs a random donor. Report results/analysis/exp104_report.txt
"""
import glob, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import gt_analysis as A_
import rank_sweep as R
from subset_probe import TARGETS as WEAK
SIGN = {"ci": -1, "dci": -1, "corr": +1, "rc": -1, "drc": -1, "cf": +1, "sel": +1, "vinn": +1, "rp_succ": -1, "rp_fail": +1}

def load():
    E = {}
    for f in sorted(glob.glob(str(REPO / "results/analysis/exp104_behav_s*.npz"))) + [str(REPO / "results/analysis/exp104_behav.npz")]:
        if not pathlib.Path(f).exists(): continue
        z = np.load(f)
        for k in SIGN:
            x = z[k]; E[k] = x.copy() if k not in E else np.where(np.isnan(E[k]), x, E[k])
    return E

def main():
    E = load(); UN = A_.UNIQUE
    T = [t for t in UN if t in WEAK and not np.isnan(E["ci"][t, [d for d in UN if d != t]]).any()]
    GT = {t: {d: A_.rec(t, d)["success_rate"] for d in UN if d != t} for t in T}
    base = {t: R.base(t)[0] for t in T}
    L = [f"=== EXP-104 behavioural signals vs ground truth: {len(T)}/15 weak receivers complete (ceiling 0.80) ===",
         f"{'signal':9s} {'sign':>4s} {'rho':>7s} {'p':>6s} {'pos':>5s} | top-3 gain vs target alone (pp) | random donor"]
    for k, sg in SIGN.items():
        rr, top, rnd = [], [], []
        for t in T:
            ds = [d for d in sorted(GT[t]) if np.isfinite(E[k][t, d])]
            if len(ds) < 20: continue
            y = np.array([GT[t][d] for d in ds]); x = sg * E[k][t, ds]
            if x.std() > 0 and y.std() > 0: rr.append(stats.spearmanr(x, y).statistic)
            top.append(100 * (np.mean(y[np.argsort(-x)[:3]]) - base[t])); rnd.append(100 * (y.mean() - base[t]))
        if not rr: continue
        rr = np.array(rr); p = stats.ttest_1samp(rr, 0).pvalue if len(rr) > 1 else np.nan
        L.append(f"{k:9s} {sg:+4d} {rr.mean():+7.3f} {p:6.3f} {int((rr>0).sum()):>2d}/{len(rr):<2d} | {np.mean(top):+6.1f} | {np.mean(rnd):+6.1f}")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp104_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
