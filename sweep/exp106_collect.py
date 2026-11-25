#!/usr/bin/env python3
"""EXP-106 stage A — equal-budget comparison of two ways to spend 80 rollouts on a target (15 weak targets).
All collection rollouts use LIBERO initial states 25..49; every final score uses states 0..24 (never seen).

  self   the ORIGINAL target LoRA, 80 episodes cycling over states 25..49, full-trajectory dumps (frames every 2
         steps + every executed action) -> data for self-imitation fine-tuning (filtered BC) in stage B.
         Its first 10 episodes (states 25..34) double as the target's own 10 rollouts for the probe's guard.
  probe  the deployed probe's donor trials: 7 seeded-random donors x 10 episodes on states 25..34, merged with
         the probe operator W = S16 + r_t + c * r_d. With the target's 10, the probe also spends 80.
Records e106_t{t}_self, e106_t{t}_pd{d}. Env: E106_MODE (self | probe), E106_TARGETS (comma list).
"""
import json, os, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
from subset_probe import TARGETS as WEAK
UNIQUE = [t for t in range(76) if t not in (18, 62)]
DUP = {7: 18, 18: 7, 29: 62, 62: 29}
MODE = os.environ.get("E106_MODE", "self")
TGTS = [int(x) for x in os.environ["E106_TARGETS"].split(",")] if os.environ.get("E106_TARGETS") else list(WEAK)
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))

def probe_donors(t):
    pool = [d for d in UNIQUE if d != t and d != DUP.get(t, -1)]
    return sorted(int(d) for d in np.random.default_rng(10600 + t).choice(pool, 7, replace=False))

def run(jobs, extra, tag):
    for j in jobs: j["suite"] = "libero_90"; j["norm_stats"] = U.FLEET_NORM
    planp = REPO / f"sweep/_e106_{tag}_{os.getpid()}.json"
    json.dump(dict(U.META, **extra, jobs=jobs, eval_script="examples/libero/main_dense.py", save_videos="0",
                   heavy_parallel=2), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)

def main():
    U._clear_orphaned_builds()
    if MODE == "self":
        jobs = [{"name": f"e106_t{t}_self", "method": "average", "params": {}, "adapters": [],
                 "adapter_paths": [U.afile[t]], "eval_scenes": [SC[t][0]]}
                for t in TGTS if U.get(f"e106_t{t}_self", SC[t][0]) is None]
        print(f"[e106] self: {len(jobs)} jobs", flush=True)
        if jobs: run(jobs, dict(n_trials=80, state_offset=25, state_pool=25,
                                dump_obs_dir=str(REPO / "results/exp106_dumps"), dump_stride=2), "self")
    else:                                               # builds ~0.9 GB each, shiva quota 46 GB: one target (7 builds) per chunk
        todo = [(t, d) for t in TGTS for d in probe_donors(t) if U.get(f"e106_t{t}_pd{d}", SC[t][0]) is None]
        print(f"[e106] probe: {len(todo)} donor jobs", flush=True)
        for i in range(0, len(todo), 7):
            built, jobs = [], []
            for t, d in todo[i:i + 7]:
                p = U.build(f"e106_{t}_{d}", t, [(d, REF / U.resid_norm(d))]); built.append(p)
                jobs.append({"name": f"e106_t{t}_pd{d}", "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [p], "eval_scenes": [SC[t][0]]})
            run(jobs, dict(n_trials=10, state_offset=25), f"probe{i}")
            for p in built: pathlib.Path(p).unlink(missing_ok=True)
            print(f"[e106] probe chunk {i//7 + 1}/{(len(todo) + 6)//7} done", flush=True)
    print(f"[e106] {MODE} DONE", flush=True)

if __name__ == "__main__":
    main()
