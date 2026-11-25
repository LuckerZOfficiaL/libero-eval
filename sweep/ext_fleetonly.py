#!/usr/bin/env python3
"""EXP-68 — was EXP-32's external-target reversal about pool COMPOSITION or pool SIZE?

EXP-32 got +12.00pp on 9 external targets where EXP-29 got -4.44pp, but three things differed:
candidates 8 -> 16, fleet-only -> mixed, and one shared candidate set -> per-target draws. The probe
picked external donors at chance, so composition looked unlikely to be the driver -- but that was an
inference, not a measurement.

This isolates it. EXP-32 already probed all 16 candidates for every external target, so the FLEET-ONLY
top-3 can be read straight off those probes: same targets, same per-target draw, same operator, same
guard, half the pool. Committing it costs 9 evaluations and answers the question directly.
  fleet-only commit ~ mixed commit  -> size/draw was the driver, composition irrelevant
  fleet-only much worse             -> the external donors were doing the work after all
"""
import json, pathlib, re, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U

def targets():
    out = []
    for tag, key, task, suite, norm in U.targets():
        if isinstance(key, str) and "goal_expert" in key:
            out.append((tag, key, task, suite, norm))
    return out

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = []
    for tag, key, task, suite, norm in targets():
        sc = {}
        for f in pathlib.Path("results/merge").glob(f"e44_p_{tag}_*.jsonl"):
            m = re.match(rf"e44_p_{re.escape(tag)}_(\d+)$", f.stem)      # fleet donors only: numeric id
            if not m: continue
            v = U.get(f.stem, task)
            if v is not None: sc[int(m.group(1))] = v
        if len(sc) < 6 or U.get(f"e74_fleetonly_{tag}", task) is not None: continue
        base = U.get(f"e44_base_{tag}", task)
        if base is None or max(sc.values()) <= base:
            print(f"[ef] {tag}: fleet-only probe abstains (best {max(sc.values()) if sc else 0:.2f} vs base {base})", flush=True)
            continue
        todo.append((tag, key, task, suite, norm, sorted(sc, key=sc.get, reverse=True)[:3], len(sc)))
    print(f"[ef] {len(todo)} external targets to commit with FLEET-ONLY top-3", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for tag, key, task, suite, norm, top, n in todo[i:i + U.CHUNK]:
            p = U.build(f"ef_{tag}", key, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e74_fleetonly_{tag}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": suite, "_norm": norm})
        for su in sorted({j["_suite"] for j in jobs}):
            sub = [{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs if j["_suite"] == su]
            nm = next(j["_norm"] for j in jobs if j["_suite"] == su)
            U.run(sub, 25, su, nm)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[ef] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)
    print("[ef] done")

if __name__ == "__main__":
    main()
