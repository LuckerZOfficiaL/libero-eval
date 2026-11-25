#!/usr/bin/env python3
"""EXP-96 evaluation: label-free energy signals vs the EXP-95 ground-truth donor ranking.
Pre-registered direction (LiST): LOWER energy = better donor, so score = -signal. Within-target Spearman over
all 73 donors; weak (25-rollout GT), strong (10-rollout GT), all. Also the 'safe acceptance' view: does
picking the donor with the lowest energy beat a random donor, and how often does it land on a harmful one?"""
import glob, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import gt_analysis as A, rank_sweep as R
from subset_probe import TARGETS as WEAK
SIG = ["disp", "drift", "demo", "ddisp"]
fs = sorted(glob.glob(str(REPO / "results/analysis/energy_proxy_s*.npz"))) or [str(REPO / "results/analysis/energy_proxy.npz")]
E = {s: None for s in SIG}
for f in fs:
    z = np.load(f)
    for s in SIG:
        E[s] = z[s].copy() if E[s] is None else np.where(np.isnan(E[s]), z[s], E[s])
GT = {}
for t in A.UNIQUE:
    GT[t] = {d: A.rec(t, d)["success_rate"] for d in A.UNIQUE if d != t}
def rank_avg(t, ds, sigs):
    return np.mean([stats.rankdata(E[s][t, ds]) for s in sigs], 0)
L = []; P = lambda *a: L.append(" ".join(str(x) for x in a))
done = [t for t in A.UNIQUE if not np.isnan(E["disp"][t, [d for d in A.UNIQUE if d != t]]).any()]
P(f"=== EXP-96: label-free energy (LiST-style) vs ground-truth donor ranking; {len(done)}/74 receivers scored ===")
P("score = -signal (lower energy = better, pre-registered). within-target Spearman over 73 donors")
groups = (("weak", [t for t in done if t in WEAK]), ("strong", [t for t in done if t not in WEAK]), ("all", done))
P(f"{'signal':<22}" + "".join(f"{g:>22}" for g, _ in groups))
combos = {s: [s] for s in SIG}; combos["disp+drift"] = ["disp", "drift"]; combos["disp+demo"] = ["disp", "demo"]; combos["all4"] = SIG
for nm, sg in combos.items():
    row = f"{nm:<22}"
    for g, T in groups:
        rr = []
        for t in T:
            ds = sorted(GT[t]); y = np.array([GT[t][d] for d in ds]); x = -rank_avg(t, ds, sg)
            if y.std() > 0 and np.std(x) > 0: rr.append(stats.spearmanr(x, y).statistic)
        rr = np.array(rr)
        row += f"{rr.mean():>+10.3f} (p={stats.ttest_1samp(rr,0).pvalue:.2f})" if len(rr) > 2 else f"{'--':>22}"
    P(row)
P("ceilings (reliability-based): weak 0.795, strong 0.409, all 0.519")
P("\npick the single lowest-energy donor (weak receivers): gain vs baseline, vs a random donor, vs the best donor")
for nm, sg in combos.items():
    gp, gr, gb, harm = [], [], [], []
    for t in [t for t in done if t in WEAK]:
        ds = sorted(GT[t]); y = np.array([GT[t][d] for d in ds]); b, _ = R.base(t)
        pick = int(np.argmin(rank_avg(t, ds, sg)))
        gp.append(100 * (y[pick] - b)); gr.append(100 * (y.mean() - b)); gb.append(100 * (y.max() - b)); harm.append(y[pick] - b < -0.08)
    if gp: P(f"   {nm:<14} pick {np.mean(gp):+6.1f}pp | random donor {np.mean(gr):+6.1f}pp | best donor {np.mean(gb):+6.1f}pp | pick harmful (<-8pp) {100*np.mean(harm):.0f}%")
txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp96_energy_report.txt").write_text(txt + "\n")
