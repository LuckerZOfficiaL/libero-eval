#!/usr/bin/env python3
"""EXP-73 — re-measure the committed merges at 50 rollouts, to match the sharper baselines.

EXP-67 halved the noise on the baseline arm. The headline weak-target gain is a difference, so it is
only as sharp as its noisier side: commits are still 25 rollouts (SE ~7.8pp per target). Re-running the
same top-3 commits at 50 rollouts puts both arms on the same footing and roughly halves the error on the
number most likely to be quoted. Same donors, same operator -- only the evaluation budget changes.
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
    todo = [t for t in WEAK if RANK[str(t)][0][1] > target_baseline(t) and U.get(f"e79_c50_t{t}", SC[t][0]) is None]
    print(f"[tc] {len(todo)} commits to re-measure at 50 rollouts", flush=True)
    for i in range(0, len(todo), max(3, U.CHUNK // 2)):
        jobs, built = [], []
        for t in todo[i:i + max(3, U.CHUNK // 2)]:
            top = [int(p) for p, _ in RANK[str(t)][:3]]
            p = U.build(f"tc_{t}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e79_c50_t{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=50, jobs=jobs), open(REPO / "sweep/_tc.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_tc.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[tc] chunk {i//max(3,U.CHUNK//2)+1} done", flush=True)
    print("[tc] done")

if __name__ == "__main__":
    main()
