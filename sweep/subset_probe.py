#!/usr/bin/env python3
"""EXP-78 -- probe SUBSETS instead of singletons, at identical rollout cost.

Every probe we have run scores donors ONE AT A TIME (target + one donor at full dose c), ranks them,
and commits the top-3 at c/3 each. That pipeline assumes singleton scores ADD. Two results say the
assumption is shaky: two independent probes of the same 8 candidates agree on the single best donor
for only 1 of 10 targets, and the (target,donor) pair interaction is real and reproducible at rho=0.58.
If interaction matters, the quantity we should be measuring is the performance of a 3-SET, not of a
singleton -- and a singleton probe never observes it.

Design (cost-matched exactly to the standard probe, 8x10 + 25 = 105 rollouts per target-draw):
for each target and each candidate draw, take the SAME 8 candidates the standard arm saw, form 8
BALANCED 3-subsets (each donor appears in exactly 3 of the 8), probe each subset at 10 rollouts
MERGED AT THE COMMITTED DOSE c/3, apply the same abstention guard, and commit the best subset @25.
The subset probe therefore evaluates exactly the object that gets deployed.

Comparison arm is free: e76_t{t}_d{k} is the standard method's commit for the SAME target and the
SAME 8-candidate menu (EXP-69/74). Paired difference = subset-probe gain - singleton-probe gain.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
from draw_variance import TARGETS as T5, cands
from draw_variance2 import TARGETS as T10

TARGETS = T5 + T10                      # the 15 weak fleet targets
DRAWS = [1, 2]                          # k-major: draw 1 completes before draw 2 starts

def subsets(t, k, cs):
    """8 triples over the 8 candidates, each candidate used exactly 3 times, none repeated in a triple."""
    rng = np.random.default_rng(90000 + 100 * t + k)
    for _ in range(10000):
        pool = np.repeat(np.arange(8), 3); rng.shuffle(pool)
        tri = [sorted(pool[i:i + 3].tolist()) for i in range(0, 24, 3)]
        if all(len(set(x)) == 3 for x in tri):
            return [sorted(int(cs[i]) for i in x) for x in tri]
    raise RuntimeError("balanced design not found")

def main():
    # 09-16: user dequeued EXP-78 in favour of EXP-81/83. The chain still calls this script, so exit
    # cleanly rather than editing the running chain (never edit a bash script while it executes).
    if (REPO / "sweep/_exp78_disabled").exists():
        print("[sub] DISABLED by sweep/_exp78_disabled -- skipping (53/240 probes retained, resumable)", flush=True)
        return
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    dose = lambda c: (ref / U.resid_norm(c)) / 3      # committed dose, used by BOTH probe and commit

    todo = [(t, k, j, s) for k in DRAWS for t in TARGETS
            for j, s in enumerate(subsets(t, k, cands(t, k)))
            if U.get(f"e86_t{t}_d{k}_s{j}", SC[t][0]) is None]
    print(f"[sub] stage 1: {len(todo)} subset probes ({len(TARGETS)} targets x {len(DRAWS)} draws x 8)", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, k, j, s in todo[i:i + U.CHUNK]:
            p = U.build(f"sub_{t}_{k}_{j}", t, [(c, dose(c)) for c in s]); built.append(p)
            jobs.append({"name": f"e86_t{t}_d{k}_s{j}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=10, jobs=jobs), open(REPO / "sweep/_sub.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_sub.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[sub] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)

    todo2 = []
    for k in DRAWS:
        for t in TARGETS:
            ss = subsets(t, k, cands(t, k))
            sc = {j: U.get(f"e86_t{t}_d{k}_s{j}", SC[t][0]) for j in range(8)}
            sc = {a: b for a, b in sc.items() if b is not None}
            if len(sc) < 6 or U.get(f"e87_t{t}_d{k}", SC[t][0]) is not None: continue
            if max(sc.values()) <= target_baseline(t):
                print(f"[sub] t{t} draw {k}: guard abstains", flush=True); continue
            todo2.append((t, k, ss[max(sc, key=sc.get)]))
    print(f"[sub] stage 2: {len(todo2)} commits", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for t, k, top in todo2[i:i + U.CHUNK]:
            p = U.build(f"subc_{t}_{k}", t, [(c, dose(c)) for c in top]); built.append(p)
            jobs.append({"name": f"e87_t{t}_d{k}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_subc.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_subc.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[sub] done")

if __name__ == "__main__":
    main()
