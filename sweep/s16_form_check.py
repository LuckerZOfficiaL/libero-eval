#!/usr/bin/env python3
"""EXP-63 — is the operator's zero-donor form worth anything by itself, measured in ONE run?

W = S16 + r_target with no donors is not the raw target adapter D_target: S16 is the fleet soup's rank-16
component, not the projection of this target. EXP-62 hinted the reconstruction scores ~7pp above the raw
adapter, but that compared runs weeks apart, and the controlled figure we already have for "protecting
the target" is 2.5pp. This settles it by evaluating BOTH arms back to back on the SAME 25 states.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline, afile
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
WEAK = [int(t) for t in RANK if target_baseline(int(t)) < 0.9]

def main():
    U._clear_orphaned_builds()
    todo = [t for t in WEAK if U.get(f"e67_raw_t{t}", SC[t][0]) is None or U.get(f"e67_s16_t{t}", SC[t][0]) is None]
    print(f"[s16] {len(todo)} targets x 2 arms (raw adapter vs S16 + r_target), 25 rollouts each", flush=True)
    for i in range(0, len(todo), max(3, U.CHUNK // 2)):
        jobs, built = [], []
        for t in todo[i:i + max(3, U.CHUNK // 2)]:
            if U.get(f"e67_s16_t{t}", SC[t][0]) is None:
                p = U.build(f"s16_{t}", t, []); built.append(p)
                jobs.append({"name": f"e67_s16_t{t}", "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [p], "eval_scenes": SC[t]})
            if U.get(f"e67_raw_t{t}", SC[t][0]) is None:      # the unmodified target LoRA, same states
                jobs.append({"name": f"e67_raw_t{t}", "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [str(afile[t])], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_s16.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(REPO / "sweep/_s16.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[s16] chunk {i//max(3,U.CHUNK//2)+1} done", flush=True)
    print("[s16] done")

if __name__ == "__main__":
    main()
