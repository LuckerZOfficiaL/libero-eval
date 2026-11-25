#!/usr/bin/env python3
"""EXP-51 — deploy the cheap probe end-to-end: 8 candidates x 5 rollouts, half the current budget.

Two independent estimates say 5 rollouts per candidate keeps most of the selection value (EXP-47:
77% of the gain at 50% of the cost, selecting on one probe run and valuing on another; EXP-22.5's
deployed curve: 65%). Both are estimates. This runs the cheap probe for real on the 15 weak fleet
targets -- fresh 5-rollout probes of the same 8 candidates, its own top-3, committed at 25 rollouts --
so the operating point can be quoted as a measured deployment number rather than a projection.
Reference arm: the existing 10-rollout probe and its commit (e27), same targets, same candidates.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
WEAK = [int(t) for t in RANK if target_baseline(int(t)) < 0.9]

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(t, int(p)) for t in WEAK for p, _ in RANK[str(t)] if U.get(f"e60_t{t}_p{p}", SC[t][0]) is None]
    print(f"[cheap] stage 1: {len(todo)} probes at 5 rollouts", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, d in todo[i:i + U.CHUNK]:
            p = U.build(f"cp_{t}_{d}", t, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e60_t{t}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=5, jobs=jobs), open(REPO / "sweep/_cp.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_cp.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[cheap] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    # stage 2: commit each target's cheap-probe top-3 at 25 rollouts, under the same guard
    todo2 = []
    for t in WEAK:
        sc = {int(p): U.get(f"e60_t{t}_p{p}", SC[t][0]) for p, _ in RANK[str(t)]}
        sc = {k: v for k, v in sc.items() if v is not None}
        if len(sc) < 4 or U.get(f"e61_commit_t{t}", SC[t][0]) is not None: continue
        best = max(sc.values())
        if best <= target_baseline(t):
            print(f"[cheap] t{t}: best 5-rollout probe {100*best:.0f}% <= base {100*target_baseline(t):.0f}% -> ABSTAIN", flush=True)
            continue
        todo2.append((t, sorted(sc, key=sc.get, reverse=True)[:3]))
    print(f"[cheap] stage 2: {len(todo2)} commits at 25 rollouts", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for t, top in todo2[i:i + U.CHUNK]:
            p = U.build(f"cpc_{t}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e61_commit_t{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_cpc.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_cpc.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[cheap] done")

if __name__ == "__main__":
    main()
