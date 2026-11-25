#!/usr/bin/env python3
"""EXP-101 report: protected whole-pool merge vs baseline, the no-donor operator, and a random single donor.

Per target t (15 weak):
  base   = target adapter alone (R.base: mean of the standing baseline runs)
  k16    = operator with no donor, S16 + r_t (e94_t{t}_k16)
  rand   = mean over the 73 single donors of the ground-truth matrix (e98) = expected value of a random donor
  best   = best single donor in the ground-truth matrix (oracle, needs 73 x 25 rollouts)
  u1/p1/p3 = EXP-101 arms (e101_t{t}_{arm})
Writes results/analysis/exp101_report.txt.
"""
import json, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from reinforce_train import scenes as SC
import unified_pool_driver as U
import rank_sweep as R
from gt_analysis import rec, UNIQUE
from subset_probe import TARGETS as WEAK

ARMS = ["u1", "p1", "p3"]

def main(targets=None):
    T = list(targets or WEAK)
    diag = {}
    for f in ("exp101_diag.json", "exp101_diag_shiva.json"):
        p = REPO / "results/analysis" / f
        if p.exists(): diag.update(json.load(open(p)))
    rows = []
    for t in T:
        r = dict(t=t, base=R.base(t)[0], k16=U.get(f"e94_t{t}_k16", SC[t][0]))
        g = [rec(t, d) for d in UNIQUE if d != t]; g = [x["success_rate"] for x in g if x is not None]
        r["rand"] = float(np.mean(g)) if g else None; r["best"] = float(np.max(g)) if g else None; r["ngt"] = len(g)
        for a in ARMS: r[a] = U.get(f"e101_t{t}_{a}", SC[t][0])
        d = diag.get(str(t), {}); r["kept"] = d.get("energy_retained"); r["ratio"] = d.get("ratio")
        rows.append(r)
    L = []; P = lambda *a: L.append(" ".join(str(x) for x in a))
    f = lambda v: "  -  " if v is None else f"{100*v:5.1f}"
    P(f"=== EXP-101 protected whole-pool merge: {len(T)} targets, 25 rollouts per arm ===")
    P(f"{'t':>4} {'base':>5} {'k16':>5} {'rand':>5} {'best':>5} | {'u1':>5} {'p1':>5} {'p3':>5} | kept  out-change")
    for r in rows:
        P(f"{r['t']:>4} {f(r['base'])} {f(r['k16'])} {f(r['rand'])} {f(r['best'])} | {f(r['u1'])} {f(r['p1'])} {f(r['p3'])} | "
          + (f"{100*r['kept']:4.1f}% {r['ratio']:.1e}" if r['kept'] is not None else ""))
    P("\nPooled over targets with every column present (pp, paired over targets):")
    cols = ["base", "k16", "rand", "best"] + ARMS
    full = [r for r in rows if all(r[c] is not None for c in cols)]
    P(f"   n = {len(full)} targets")
    if full:
        M = {c: np.array([r[c] for r in full]) * 100 for c in cols}
        P("   mean success: " + "  ".join(f"{c} {M[c].mean():.1f}" for c in cols))
        for a in ARMS:
            for ref in ("base", "k16", "rand"):
                dlt = M[a] - M[ref]
                p = stats.wilcoxon(dlt).pvalue if np.any(dlt != 0) and len(dlt) > 1 else float("nan")
                P(f"   {a} - {ref:>4}: {dlt.mean():+6.2f} pp  (wins {int((dlt>0).sum())}/{len(dlt)}, losses {int((dlt<0).sum())}, Wilcoxon p={p:.3f})")
        for a, b in (("p1", "u1"), ("p3", "p1"), ("p3", "u1")):
            dlt = M[a] - M[b]
            p = stats.wilcoxon(dlt).pvalue if np.any(dlt != 0) and len(dlt) > 1 else float("nan")
            P(f"   {a} - {b:>4}: {dlt.mean():+6.2f} pp  (wins {int((dlt>0).sum())}/{len(dlt)}, Wilcoxon p={p:.3f})")
        # share of targets where the arm is harmful vs base (< -8pp) / helpful (> +8pp), compared with a random donor
        for a in ARMS:
            d = M[a] - M["base"]
            P(f"   {a}: helpful (>+8pp vs base) {int((d>8).sum())}/{len(d)}, harmful (<-8pp) {int((d<-8).sum())}/{len(d)}")
    out = REPO / "results/analysis/exp101_report.txt"; out.write_text("\n".join(L) + "\n"); print("\n".join(L))

if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else None)
