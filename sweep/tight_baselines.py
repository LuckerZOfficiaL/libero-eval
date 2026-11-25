#!/usr/bin/env python3
"""EXP-67 (odin) — halve the noise on the numbers we cite most: 50-rollout baselines.

EXP-63 showed identical weights on identical states move a single target by up to 30pp between runs.
Every headline is a difference against a 25-rollout baseline, so that baseline carries ~10pp of standard
error per target and is the noisiest term in most comparisons. Re-measuring the 15 weak fleet targets at
50 rollouts (states 0-49) roughly halves it, sharpening every future paired test on those targets.
Deliberately given to odin: it is pure evaluation, needs no fast GPU, and leaves shiva for selection work.
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
    todo = [t for t in WEAK if U.get(f"e73_base50_t{t}", SC[t][0]) is None]
    print(f"[tb] {len(todo)} targets x 50 rollouts (raw target adapter)", flush=True)
    for i in range(0, len(todo), max(3, U.CHUNK)):
        jobs = [{"name": f"e73_base50_t{t}", "method": "average", "params": {}, "adapters": [],
                 "adapter_paths": [str(afile[t])], "eval_scenes": SC[t]} for t in todo[i:i + max(3, U.CHUNK)]]
        json.dump(dict(U.META, n_trials=50, jobs=jobs, eval_script="examples/libero/main_dense.py"),
                  open(REPO / "sweep/_tb.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_tb.json")], cwd=REPO)
        print(f"[tb] chunk {i//max(3,U.CHUNK)+1} done", flush=True)
    print("[tb] done")

if __name__ == "__main__":
    main()
