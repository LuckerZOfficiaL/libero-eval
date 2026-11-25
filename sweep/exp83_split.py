#!/usr/bin/env python3
"""EXP-83 step 1: the 80/20 donor/receiver split, balanced across (provenance x difficulty).

Proportional 20% would put 12 of 18 receivers at ceiling, where the guard abstains and nothing is
measurable. Receivers need headroom; donors do not. So the RECEIVER set is balanced across the 2x2
and the donor pool simply keeps everything else. Frozen to a file so every later stage reads the
same split (a re-drawn split between training and evaluation would silently invalidate the result).
"""
import json, pathlib, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U

OUT = REPO / "results/analysis/exp83_split.json"
PLAN = {("fleet", "weak"): 7, ("fleet", "strong"): 5, ("ext", "weak"): 3, ("ext", "strong"): 3}

def main():
    rows = []
    for tag, tk, task, suite, norm in U.targets():
        b = U.get(f"e44_base_{tag}", task)
        if b is None: continue
        # key must be exactly what U.fold()/U.EXTPATH accept: an int for fleet, the BARE adapter name
        # for external. An "np:" prefix here produced KeyError: np:pi05_libero_goal_expert (09-18).
        rows.append(dict(tag=tag, key=tk, task=task, suite=suite,
                         base=b, fleet=isinstance(tk, int)))
    strata = {}
    for r in rows:
        strata.setdefault(("fleet" if r["fleet"] else "ext", "weak" if r["base"] < 0.9 else "strong"), []).append(r)
    rng = np.random.default_rng(83)
    recv = []
    for k, n in sorted(PLAN.items()):
        pool = sorted(strata.get(k, []), key=lambda r: r["tag"])
        if len(pool) < n: raise RuntimeError(f"stratum {k} has {len(pool)} < {n}")
        for i in sorted(rng.choice(len(pool), n, replace=False)): recv.append(pool[i])
    rtags = {r["tag"] for r in recv}
    donors = [r for r in rows if r["tag"] not in rtags]
    out = dict(receivers=recv, donors=donors,
               note="receivers balanced across provenance x difficulty; donors = everything else")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"[split] {len(donors)} donors / {len(recv)} receivers -> {OUT.name}")
    for k in sorted(PLAN):
        got = [r for r in recv if ("fleet" if r["fleet"] else "ext") == k[0] and ("weak" if r["base"] < 0.9 else "strong") == k[1]]
        print(f"  receivers {k[0]:5s}/{k[1]:6s}: {len(got)}  {[r['tag'] for r in got]}")
    print(f"  donor pool: {sum(1 for d in donors if d['fleet'])} fleet + {sum(1 for d in donors if not d['fleet'])} external")
    print(f"  receiver baselines: weak {np.mean([r['base'] for r in recv if r['base']<0.9])*100:.1f}%, "
          f"strong {np.mean([r['base'] for r in recv if r['base']>=0.9])*100:.1f}%")

if __name__ == "__main__":
    main()
