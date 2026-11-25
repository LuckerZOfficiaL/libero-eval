#!/usr/bin/env python3
"""EXP-74 — draw-averaged headline: extend EXP-69's replication to the remaining weak fleet targets.

Every number we quote uses ONE candidate draw per target, and EXP-69 measured the within-target spread
across draws at 13.6pp. With per-target draws that noise is independent and sits inside our across-target
SE, so the headline is not biased -- but it is noisier than it needs to be, and a draw-averaged estimate
is what should go in the paper. EXP-69 covered 5 targets with 4 draws each; this covers the other 10
weak fleet targets with 3 draws each, giving a draw-averaged gain across all 15.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
from draw_variance import TARGETS as DONE
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
TARGETS = [int(t) for t in RANK if target_baseline(int(t)) < 0.9 and int(t) not in DONE]
DRAWS = [1, 2, 3]

def cands(t, k):
    rng = np.random.default_rng(70000 + 100 * t + k)
    return sorted(int(x) for x in rng.choice([c for c in range(76) if c != t], 8, replace=False))

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(t, k, d) for t in TARGETS for k in DRAWS for d in cands(t, k)
            if U.get(f"e75_t{t}_d{k}_p{d}", SC[t][0]) is None]
    print(f"[dv2] {len(TARGETS)} targets x {len(DRAWS)} draws; stage 1: {len(todo)} probes", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, k, d in todo[i:i + U.CHUNK]:
            p = U.build(f"dv2_{t}_{k}_{d}", t, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e75_t{t}_d{k}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=10, jobs=jobs), open(REPO / "sweep/_dv2.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_dv2.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[dv2] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    todo2 = []
    for t in TARGETS:
        for k in DRAWS:
            sc = {d: U.get(f"e75_t{t}_d{k}_p{d}", SC[t][0]) for d in cands(t, k)}
            sc = {a: x for a, x in sc.items() if x is not None}
            if len(sc) < 6 or U.get(f"e76_t{t}_d{k}", SC[t][0]) is not None: continue
            if max(sc.values()) <= target_baseline(t):
                print(f"[dv2] t{t} draw {k}: guard abstains", flush=True); continue
            todo2.append((t, k, sorted(sc, key=sc.get, reverse=True)[:3]))
    print(f"[dv2] stage 2: {len(todo2)} commits", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for t, k, top in todo2[i:i + U.CHUNK]:
            p = U.build(f"dv2c_{t}_{k}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e76_t{t}_d{k}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_dv2c.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_dv2c.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[dv2] commit chunk {i//U.CHUNK+1}/{(len(todo2)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[dv2] done")

if __name__ == "__main__":
    main()
