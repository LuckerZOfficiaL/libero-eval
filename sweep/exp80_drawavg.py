#!/usr/bin/env python3
"""EXP-80 draw-averaging: blind-pool (84 adapters, draws 1-4) vs fleet-only (76, e76 draws 1-3) on the
14 weak LIBERO-90 receivers, per target, draw-averaged. Abstention (no commit record) = baseline = 0 gain."""
import os, sys, pathlib, numpy as np
os.environ["ONLY_WEAK_FLEET"] = "1"
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U, blind_pool as B
from reinforce_train import scenes as SC, target_baseline
from scipy import stats
rows = []
for tag, tk, task, suite, norm, b, cs, need in B.plan():
    blind = []
    for sfx in ("", "d2", "d3", "d4"):
        probes = [U.get(f"e89{sfx}_p_{tag}_{B.key(c)}", task) for c in B.cands(tag, tk, [*range(76), *sorted(U.EXTPATH)]) if sfx == "" or True]
        c = U.get(f"e89{sfx}_c_{tag}", task)
        if c is not None: blind.append(c - b)
        elif sfx in ("", "d2", "d3"): blind.append(0.0)          # draws 1-3 complete: absent = abstained
    fb = target_baseline(tk)
    fleet = [U.get(f"e76_t{tk}_d{k}", SC[tk][0]) for k in (1, 2, 3)]
    has_fleet = any(v is not None for v in fleet)          # targets outside EXP-69/74's weak-15 have no fleet-only draws
    fleet = [(v if v is not None else fb) - fb for v in fleet]
    rows.append((tk, b, np.mean(blind), len(blind), np.mean(fleet) if has_fleet else np.nan, blind, fleet))
    print(f"t{tk:2d} base {b*100:3.0f}%  blind {100*np.mean(blind):+6.2f} ({len(blind)} draws: {' '.join(f'{100*x:+.0f}' for x in blind)})   "
          f"fleet-only {100*np.mean(fleet):+6.2f} ({' '.join(f'{100*x:+.0f}' for x in fleet)})")
bl = np.array([r[2] for r in rows]); n = len(rows)
print(f"\nn={n} weak LIBERO-90 receivers, draw-averaged blind pool (84): {100*bl.mean():+.2f}pp  SE {100*bl.std(ddof=1)/np.sqrt(n):.2f}   improving {(bl>0).sum()}/{n}")
ok = [r for r in rows if not np.isnan(r[4])]; bl = np.array([r[2] for r in ok]); fl = np.array([r[4] for r in ok]); d = bl - fl; n = len(ok)
print(f"paired on the {n} targets that also have fleet-only draws (EXP-69/74):")
print(f"  blind pool (84)   {100*bl.mean():+.2f}pp  SE {100*bl.std(ddof=1)/np.sqrt(n):.2f}   improving {(bl>0).sum()}/{n}")
print(f"  fleet-only (76)   {100*fl.mean():+.2f}pp  SE {100*fl.std(ddof=1)/np.sqrt(n):.2f}   improving {(fl>0).sum()}/{n}")
print(f"  paired blind - fleet  {100*d.mean():+.2f}pp  SE {100*d.std(ddof=1)/np.sqrt(n):.2f}  p={stats.ttest_rel(bl,fl).pvalue:.3f}   (single-draw EXP-80 was -5.19, p=0.075)")
sd_within = np.sqrt(np.mean([np.var(r[5], ddof=1) for r in rows if len(r[5])>1])); print(f"  within-target draw SD (blind): {100*sd_within:.1f}pp")
