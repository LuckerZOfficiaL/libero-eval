#!/usr/bin/env python3
"""ROUTES 2 and 3 validation — record PROGRESS and SUCCESS_STEP for probe merges at high n.

Re-probes 3 weak fleet targets x their 8 candidates at 25 rollouts using the instrumented eval script,
so every episode yields (success, progress, success_step) instead of just success.

That single run answers both:
  route 2: is a dense score cheaper than binary? Compare how well mean-progress over the first k
           episodes predicts the 25-episode success rate, against the success rate over the same k.
           If progress@3 matches binary@10, the probe gets ~3x cheaper AND every learned selector's
           ceiling rises, since sqrt(0.41)=0.64 is a label-noise limit, not a model limit.
  route 3: where can an episode be truncated? Successes break early, failures always run to max_steps
           (220-520 by suite), so the success-step distribution says what a cutoff would cost.
"""
import json, pathlib, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline

TARGETS = [6, 50, 56]
ROLLOUTS = 25
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())

if __name__ == "__main__":
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(t, int(p)) for t in TARGETS for p, _ in RANK[str(t)]
            if not (REPO / f"results/merge/e58_t{t}_p{p}.jsonl").exists()]
    print(f"[dense] {len(todo)} probe merges at {ROLLOUTS} rollouts (targets {TARGETS})", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, d in todo[i:i + U.CHUNK]:
            p = U.build(f"dense_{t}_{d}", t, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e58_t{t}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        planp = REPO / "sweep/_dense_current.json"
        json.dump(dict(U.META, n_trials=ROLLOUTS, jobs=jobs, suite="libero_90", norm_stats=U.FLEET_NORM,
                       eval_script="examples/libero/main_dense.py"), open(planp, "w"), indent=1)
        import subprocess
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
        print(f"[dense] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)
    print("[dense] done")
