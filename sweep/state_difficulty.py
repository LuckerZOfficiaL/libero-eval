#!/usr/bin/env python3
"""EXP-57 — map LIBERO's per-state difficulty, because the probe and the commit do not share states.

LIBERO replays fixed initial states (initial_states[episode_idx], seed 7), so a 10-rollout probe always
scores states 0-9 and a 25-rollout commit scores 0-24. On the 24 instrumented merges of EXP-48, states
0-9 came out 11.5pp EASIER than 10-24 for every candidate, selected or not. If that holds generally it
means the guard -- "best probe (states 0-9) > baseline (states 0-24)" -- is optimistic by construction,
which would explain committed merges that harm despite passing, and most of the +5pp "winner's curse".

This measures the difficulty profile directly: the unmerged target adapter (the baseline itself) on 50
episodes for the weak fleet targets, instrumented so every episode's outcome is recorded per state.
Cheap and decisive: if the profile is flat the concern dies; if states 0-9 are easy, the fix is to score
probe, baseline and commit on the SAME states.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
WEAK = [int(t) for t in RANK if target_baseline(int(t)) < 0.9]
N = 50

def main():
    U._clear_orphaned_builds()
    todo = [t for t in WEAK if U.get(f"e63_base_t{t}", SC[t][0]) is None]
    print(f"[state] {len(todo)} targets x {N} episodes, instrumented (per-state outcomes)", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t in todo[i:i + U.CHUNK]:
            p = U.build(f"sb_{t}", t, []); built.append(p)      # target alone = S16 + r_target = D_target
            jobs.append({"name": f"e63_base_t{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=N, jobs=jobs, eval_script="examples/libero/main_dense.py"),
                  open(REPO / "sweep/_state.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(REPO / "sweep/_state.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[state] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)
    print("[state] done")

if __name__ == "__main__":
    main()
