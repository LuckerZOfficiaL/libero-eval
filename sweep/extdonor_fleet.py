#!/usr/bin/env python3
"""EXP-79 -- the missing cell: EXTERNAL donors into LIBERO-90 receivers.

The provenance cross-product has four cells. Three are measured:
  fleet donors  -> fleet receivers    the core method, +2.22pp over all 76
  fleet donors  -> external receivers EXP-68/75, +5.33pp draw-averaged
  both          -> either             EXP-32 union, +2.82pp over 85
  external donors -> fleet receivers  NEVER COMMITTED -- only ever available as part of a mixed pool.
We know the probe REJECTS external donors for fleet targets (20 of 228 top-3 slots, 8.8%, against a 50%
chance share, p=1.3e-40) -- a far stronger rejection than the indifference it shows on external targets
(44.4%, p=0.7). But "the probe declines to pick them" is not the same measurement as "committing them
hurts", and only the latter closes the cell.

Design, exactly symmetric to EXP-68: EXP-32 already probed all 8 external candidates against every fleet
target, so the EXTERNAL-ONLY top-3 can be read straight off those probes. Same targets, same operator
(form chosen by the target's own capture@16, as build() does), same c/3 dose, same guard, same 25-rollout
commit. Only the donors' provenance changes. Costs one commit per fleet target and nothing else.

Paired against BOTH published fleet-receiver arms: +2.22pp (fleet-only candidates) and +1.74pp (mixed).
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC

def main():
    U._clear_orphaned_builds()
    ext = sorted(U.EXTPATH)
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo, abst = [], 0
    for t in range(76):
        task = SC[t][0]
        if U.get(f"e88_f{t}", task) is not None: continue
        base = U.get(f"e44_base_f{t}", task)
        if base is None: continue
        sc = {n: U.get(f"e44_p_f{t}_{n[11:]}", task) for n in ext}
        sc = {a: b for a, b in sc.items() if b is not None}
        if len(sc) < 6: continue
        if max(sc.values()) <= base:            # same abstention guard as every other arm
            abst += 1; continue
        todo.append((t, sorted(sc, key=sc.get, reverse=True)[:3]))
    print(f"[extdon] {len(todo)} commits, {abst} guard-abstentions, over 76 fleet targets", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, top in todo[i:i + U.CHUNK]:
            p = U.build(f"xd_{t}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e88_f{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_xd.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_xd.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[extdon] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[extdon] done")

if __name__ == "__main__":
    main()
