#!/usr/bin/env python3
"""EXP-65 — at a FIXED 80-rollout budget, is 16 candidates x 5 better than 8 x 10?

EXP-47b found breadth beats depth 2.2x at equal budget, but that was a label-space simulation confined
to the 8 candidates we happened to have, so it could only re-slice rollouts, never widen the pool. The
claim that matters is the one it could not test: given the SAME 80 rollouts per target, does probing
twice as many candidates half as deeply select better donors? Good donors are rare (1-2 of 8), so
covering more of the pool should matter more than ranking a few precisely -- that is the mechanism the
simulation implied and this measures it directly.

8 fresh candidates per target (disjoint from the original 8, same seeded-random draw rule), probed at 5
rollouts; the original 8 already have 5-rollout probes from EXP-51. Then commit the top-3 of all 16 at
25 rollouts and compare with the existing 8x10 commit.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
WEAK = [int(t) for t in RANK if target_baseline(int(t)) < 0.9]

def extra_candidates(t):
    """8 more donors, drawn like the originals but excluding them, so the 16 are one honest pool."""
    have = {int(p) for p, _ in RANK[str(t)]}
    rng = np.random.default_rng(5000 + t)
    pool = [c for c in range(76) if c != t and c not in have]
    return sorted(int(x) for x in rng.choice(pool, 8, replace=False))

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(t, d) for t in WEAK for d in extra_candidates(t) if U.get(f"e70_t{t}_p{d}", SC[t][0]) is None]
    print(f"[br] stage 1: {len(todo)} extra probes at 5 rollouts", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, d in todo[i:i + U.CHUNK]:
            p = U.build(f"br_{t}_{d}", t, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e70_t{t}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=5, jobs=jobs), open(REPO / "sweep/_br.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_br.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[br] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    todo2 = []
    for t in WEAK:
        sc = {}
        for d, _ in RANK[str(t)]:
            v = U.get(f"e60_t{t}_p{d}", SC[t][0])          # original 8 at 5 rollouts (EXP-51)
            if v is not None: sc[int(d)] = v
        for d in extra_candidates(t):
            v = U.get(f"e70_t{t}_p{d}", SC[t][0])
            if v is not None: sc[d] = v
        if len(sc) < 12 or U.get(f"e71_commit_t{t}", SC[t][0]) is not None: continue
        if max(sc.values()) <= target_baseline(t):
            print(f"[br] t{t}: 16-candidate probe abstains", flush=True); continue
        todo2.append((t, sorted(sc, key=sc.get, reverse=True)[:3], len(sc)))
    print(f"[br] stage 2: {len(todo2)} commits from 16 candidates", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for t, top, n in todo2[i:i + U.CHUNK]:
            p = U.build(f"brc_{t}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e71_commit_t{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_brc.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_brc.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[br] done")

if __name__ == "__main__":
    main()
