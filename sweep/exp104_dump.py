#!/usr/bin/env python3
"""EXP-104 stage 0: target-alone rollouts (operator form, base + S16 + r_t = the policy every donor is added onto)
for the 15 weak targets, 25 episodes each, dumping both camera frames + state at EVERY replan, per episode,
tagged success / failure. These are the target's own on-policy states (incl. its failure states) for the replay
signal. No donor is tried. Records e104_t{t}_k16 (also a second, independent measurement of the no-donor operator).
Dumps: results/exp104_dumps/e104_t{t}_k16/{scene}/ep{i}_{success|failure}.npz
"""
import json, pathlib, subprocess, sys
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
import rank_sweep as R
from reinforce_train import scenes as SC
from subset_probe import TARGETS as WEAK

def main():
    U._clear_orphaned_builds()
    todo = [t for t in WEAK if U.get(f"e104_t{t}_k16", SC[t][0]) is None]
    jobs, built = [], []
    for t in todo:
        p = R.build_swap(f"e104_{t}_k16", t, 16); built.append(p)
        jobs.append({"name": f"e104_t{t}_k16", "method": "average", "params": {}, "adapters": [], "adapter_paths": [p],
                     "eval_scenes": [SC[t][0]], "suite": "libero_90", "norm_stats": U.FLEET_NORM})
    print(f"[dump] {len(jobs)} target-alone jobs", flush=True)
    planp = REPO / "sweep/_e104_dump.json"
    json.dump(dict(U.META, n_trials=U.COMMIT_R, jobs=jobs, dump_obs_dir=str(REPO / "results/exp104_dumps"),
                   save_videos="0"), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)
    for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[dump] DONE", flush=True)

if __name__ == "__main__":
    main()
