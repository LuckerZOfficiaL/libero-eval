#!/usr/bin/env python3
"""EXP-95 — GROUND-TRUTH donor matrix: every task as receiver, every other task as its single donor once.

Operator = exactly the probe's:  W = S16 + r_t + c * r_d,  c = ref / ||r_d||  (energy-matched, full dose)
74 unique fleet tasks (t18 = t7 and t62 = t29 are byte-identical copies -> dropped as targets and donors),
73 donors per target = 5,402 pairs. Evaluated with main_dense.py (per-episode success, progress, success
step) on the receiver's own task. Records e98_t{t}_d{d}. Resumable; one orchestrator at a time.

Env: GT_TARGETS = comma list of target ids (default: the 15 weak fleet targets)
     GT_N       = rollouts per pair (default 25)
"""
import json, os, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
from subset_probe import TARGETS as WEAK

UNIQUE = [t for t in range(76) if t not in (18, 62)]
TGTS = [int(x) for x in os.environ["GT_TARGETS"].split(",")] if os.environ.get("GT_TARGETS") else WEAK
NR = int(os.environ.get("GT_N", "25"))
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))

def done(t, d):
    return U.get(f"e98_t{t}_d{d}", SC[t][0]) is not None

def plan_file(jobs, k):
    for j in jobs: j["suite"] = "libero_90"; j["norm_stats"] = U.FLEET_NORM
    planp = REPO / f"sweep/_gt_{os.getpid()}_{k}.json"
    json.dump(dict(U.META, n_trials=NR, eval_script="examples/libero/main_dense.py", heavy_parallel=int(os.environ.get("GT_HEAVY", "1")), jobs=jobs), open(planp, "w"), indent=1)
    return planp

def build_chunk(chunk, keep):
    """Build a chunk's adapters. `keep` = files of the chunk currently being evaluated: their mtimes are
    refreshed before every build, because _ensure_space deletes build files older than 30 min when the
    disk is short, and with builds overlapping evaluation those files can be that old."""
    out = []
    for t, d in chunk:
        for f in keep:
            if os.path.exists(f): os.utime(f, None)
        out.append((t, d, U.build(f"gt_{t}_{d}", t, [(d, REF / U.resid_norm(d))])))
    return out

def main():
    U._clear_orphaned_builds()
    todo = [(t, d) for t in TGTS for d in UNIQUE if d != t and not done(t, d)]
    print(f"[gt] targets {TGTS} x {len(UNIQUE)-1} donors: {len(todo)} pairs to run @ {NR} rollouts; ref {REF:.3f}; chunk {U.CHUNK}", flush=True)
    chunks = [todo[i:i + U.CHUNK] for i in range(0, len(todo), U.CHUNK)]
    if not chunks: print("[gt] done", flush=True); return
    cur = build_chunk(chunks[0], [])
    for k in range(len(chunks)):
        jobs = [{"name": f"e98_t{t}_d{d}", "method": "average", "params": {}, "adapters": [],
                 "adapter_paths": [p], "eval_scenes": [SC[t][0]]} for t, d, p in cur]
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(plan_file(jobs, k))], cwd=REPO)
        nxt = build_chunk(chunks[k + 1], [p for _, _, p in cur]) if k + 1 < len(chunks) else None    # overlaps the eval
        proc.wait()
        for _, _, p in cur: pathlib.Path(p).unlink(missing_ok=True)
        cur = nxt
        n = sum(1 for t in TGTS for d in UNIQUE if d != t and done(t, d))
        print(f"[gt] {n}/{len(TGTS)*(len(UNIQUE)-1)} pairs done", flush=True)
    print("[gt] done", flush=True)

if __name__ == "__main__":
    main()
