#!/usr/bin/env python3
"""EXP-62 — are the method's gains real on initial states it never saw?

LIBERO replays fixed initial states. The probe selects donors using states 0-9 and the commit is scored
on states 0-24, so selection and scoring overlap: some of the measured +9.78pp on weak targets could be
donors that happen to suit those particular states. Nothing in the project has tested this, and it
affects every headline number we report.

Clean test: re-evaluate each committed merge on states 25-49, which no probe and no commit has ever
touched. EXP-57 already measured the unmerged targets on all 50 states, so the held-out baseline is
free and the comparison is exactly like-for-like.
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
    todo = [t for t in WEAK if RANK[str(t)][0][1] > target_baseline(t)
            and U.get(f"e66_ho_t{t}", SC[t][0]) is None]
    print(f"[ho] {len(todo)} committed merges to re-score on held-out states 25-49", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t in todo[i:i + U.CHUNK]:
            top = [int(p) for p, _ in RANK[str(t)][:3]]
            p = U.build(f"ho_{t}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e66_ho_t{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs, eval_script="examples/libero/main_dense.py",
                       state_offset=25), open(REPO / "sweep/_ho.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(REPO / "sweep/_ho.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[ho] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)
    print("[ho] done")

if __name__ == "__main__":
    main()
