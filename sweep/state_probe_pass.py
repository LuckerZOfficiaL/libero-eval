#!/usr/bin/env python3
"""EXP-84 stage 1b -- per-STATE outcomes of the singleton probe, for the item-response analysis.

An item-response probe needs, for each candidate merge, WHICH of the fixed initial states it succeeds on.
The e75 archive holds only rates. This re-probes the same (target, candidate) pairs -- the 15 weak fleet
targets x their draw-1 menus, 8 donors each, at full dose c exactly as the deployed probe does -- with
main_dense.py over all 25 fixed states, so every rollout is attributable to its state. Records
e93_t{t}_p{d}: 25 episodes with PROGRESS + SUCCESS_STEP. 15 x 8 x 25 = 3,000 rollouts.
Rates over the first 10 states must reproduce e75 within noise -- a built-in consistency check.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
from subset_probe import TARGETS
from draw_variance import cands

def main(dry=False):
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(t, d) for t in TARGETS for d in cands(t, 1) if U.get(f"e93_t{t}_p{d}", SC[t][0]) is None]
    print(f"[sp] {len(todo)} per-state probes to run ({len(TARGETS)} targets x 8 candidates x 25 states)", flush=True)
    if dry: return
    if not dry: U._clear_orphaned_builds()
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, d in todo[i:i + U.CHUNK]:
            p = U.build(f"sp_{t}_{d}", t, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e93_t{t}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, eval_script="examples/libero/main_dense.py", jobs=jobs),
                  open(REPO / "sweep/_sp.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_sp.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[sp] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[sp] done", flush=True)

if __name__ == "__main__":
    main("--dry" in sys.argv)
