#!/usr/bin/env python3
"""EXP-52 — how many donors should the commit merge? N_PICK=3 was chosen early and never validated.

The operator commits the probe's top-3 at c/3 each. Three was an initial choice, not a measurement, and
two later findings make it worth testing: good donors are rare (1-2 of 8), and the probe cannot identify
the single best one (it repeats on 1 target in 10), which argues for committing a set -- but says nothing
about whether that set should be 2, 3 or 5. Each arm uses the SAME existing 10-rollout probe ranking and
the same energy budget (coefficient c/K), so only K changes.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
WEAK = [int(t) for t in RANK if target_baseline(int(t)) < 0.9]
KS = [1, 2, 4, 5]           # K=3 already exists as e27_commit_t*_top3

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = []
    for t in WEAK:
        ranked = [int(p) for p, s in RANK[str(t)]]
        if RANK[str(t)][0][1] <= target_baseline(t): continue     # guard abstains: no commit in any arm
        for K in KS:
            if U.get(f"e62_k{K}_t{t}", SC[t][0]) is None: todo.append((t, K, ranked[:K]))
    print(f"[npick] {len(todo)} commits ({len(WEAK)} weak targets x K in {KS})", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, K, top in todo[i:i + U.CHUNK]:
            p = U.build(f"np_{K}_{t}", t, [(c, (ref / U.resid_norm(c)) / K) for c in top]); built.append(p)
            jobs.append({"name": f"e62_k{K}_t{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_np.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_np.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[npick] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[npick] done")

if __name__ == "__main__":
    main()
