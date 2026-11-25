#!/usr/bin/env python3
"""EXP-69 — how much of the method's gain is luck of the candidate draw?

EXP-68 showed the same 9 targets swing from -4.44pp to +7.56pp purely by redrawing the 8 fleet
candidates. If a single draw can move the result that far, then every number we quote from ONE draw per
target -- including the headline +8.71pp on weak fleet targets -- carries an unmeasured variance term,
and "how many candidates to probe" stops being a cost question and becomes a reliability question.

Design: 5 weak fleet targets, 3 ADDITIONAL independent candidate draws each (the existing e27 draw is
the fourth), each drawn like the original (8 fleet donors, excluding the target). Full method per draw:
probe 8 x 10, commit top-3 @25 under the guard. The spread across draws for a fixed target is the
quantity of interest; the mean across draws is a better estimate of the method than any single draw.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
TARGETS = [6, 19, 50, 15, 56]
DRAWS = [1, 2, 3]

def cands(t, k):
    rng = np.random.default_rng(70000 + 100 * t + k)
    return sorted(int(x) for x in rng.choice([c for c in range(76) if c != t], 8, replace=False))

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(t, k, d) for t in TARGETS for k in DRAWS for d in cands(t, k)
            if U.get(f"e75_t{t}_d{k}_p{d}", SC[t][0]) is None]
    print(f"[dv] stage 1: {len(todo)} probes ({len(TARGETS)} targets x {len(DRAWS)} draws x 8)", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, k, d in todo[i:i + U.CHUNK]:
            p = U.build(f"dv_{t}_{k}_{d}", t, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e75_t{t}_d{k}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=10, jobs=jobs), open(REPO / "sweep/_dv.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_dv.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[dv] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    todo2 = []
    for t in TARGETS:
        for k in DRAWS:
            sc = {d: U.get(f"e75_t{t}_d{k}_p{d}", SC[t][0]) for d in cands(t, k)}
            sc = {a: b for a, b in sc.items() if b is not None}
            if len(sc) < 6 or U.get(f"e76_t{t}_d{k}", SC[t][0]) is not None: continue
            if max(sc.values()) <= target_baseline(t):
                print(f"[dv] t{t} draw {k}: guard abstains", flush=True); continue
            todo2.append((t, k, sorted(sc, key=sc.get, reverse=True)[:3]))
    print(f"[dv] stage 2: {len(todo2)} commits", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for t, k, top in todo2[i:i + U.CHUNK]:
            p = U.build(f"dvc_{t}_{k}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e76_t{t}_d{k}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_dvc.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_dvc.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[dv] done")

if __name__ == "__main__":
    main()
