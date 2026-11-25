#!/usr/bin/env python3
"""EXP-66 — redo the 16-candidate selection with FAIR tie-breaking. No new probes needed.

EXP-65 combined 16 candidates probed at 5 rollouts and took the top 3. At 5 rollouts a score can only
be one of six values, so ties among the leaders are the norm -- and Python's sort is stable, so ties
were resolved in insertion order, which put the original 8 first every time. The new half won just
6 of 45 top-3 slots against a 50% chance rate, so the arm never really used 16 candidates.

Same probe data, ties broken at random (seeded), then commit. This is what the equal-budget breadth
test was supposed to measure.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
from breadth_test import extra_candidates
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
WEAK = [int(t) for t in RANK if target_baseline(int(t)) < 0.9]

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = []
    for t in WEAK:
        sc = {}
        for p, _ in RANK[str(t)]:
            v = U.get(f"e60_t{t}_p{p}", SC[t][0])
            if v is not None: sc[int(p)] = v
        for d in extra_candidates(t):
            v = U.get(f"e70_t{t}_p{d}", SC[t][0])
            if v is not None: sc[d] = v
        if len(sc) < 12 or U.get(f"e72_commit_t{t}", SC[t][0]) is not None: continue
        if max(sc.values()) <= target_baseline(t): continue
        rng = np.random.default_rng(9000 + t)
        jitter = {c: sc[c] + 1e-6 * rng.random() for c in sc}      # random, not insertion order
        todo.append((t, sorted(jitter, key=jitter.get, reverse=True)[:3]))
    print(f"[bf] {len(todo)} commits with fair tie-breaking", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, top in todo[i:i + U.CHUNK]:
            p = U.build(f"bf_{t}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e72_commit_t{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_bf.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_bf.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[bf] done")

if __name__ == "__main__":
    main()
