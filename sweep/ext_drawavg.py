#!/usr/bin/env python3
"""EXP-75 — draw-average the EXTERNAL-target claim, the one that most needs it.

EXP-68 showed fleet donors do help foreign targets (+7.56pp) and that the earlier -4.44pp was an
artifact of sharing ONE candidate draw across all nine. That correction is the most surprising thing in
the deck -- and it currently rests on a single draw per target, exactly the fragility that caused the
original error. Two further independent per-target draws of 8 FLEET candidates each (fleet-only, the
condition that carries the claim), full method per draw, so the external result becomes draw-averaged
like the fleet headline.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
DRAWS = [1, 2]

def ext_targets():
    return [(tag, key, task, suite, norm) for tag, key, task, suite, norm in U.targets()
            if isinstance(key, str) and "goal_expert" in key]

def cands(tag, k):
    import hashlib
    rng = np.random.default_rng(int(hashlib.md5(f"{tag}_{k}".encode()).hexdigest()[:6], 16))   # stable, not salted
    return sorted(int(x) for x in rng.choice(76, 8, replace=False))

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    TG = ext_targets()
    todo = [(tg, k, d) for tg in TG for k in DRAWS for d in cands(tg[0], k)
            if U.get(f"e80_{tg[0]}_d{k}_p{d}", tg[2]) is None]
    print(f"[ed] {len(TG)} external targets x {len(DRAWS)} draws; stage 1: {len(todo)} probes", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for (tag, key, task, suite, norm), k, d in todo[i:i + U.CHUNK]:
            p = U.build(f"ed_{tag}_{k}_{d}", key, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e80_{tag}_d{k}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": suite, "_norm": norm})
        for su in sorted({j["_suite"] for j in jobs}):
            sub = [{a: b for a, b in j.items() if not a.startswith("_")} for j in jobs if j["_suite"] == su]
            U.run(sub, 10, su, next(j["_norm"] for j in jobs if j["_suite"] == su))
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[ed] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    todo2 = []
    for tag, key, task, suite, norm in TG:
        base = U.get(f"e44_base_{tag}", task)
        if base is None: continue
        for k in DRAWS:
            sc = {d: U.get(f"e80_{tag}_d{k}_p{d}", task) for d in cands(tag, k)}
            sc = {a: x for a, x in sc.items() if x is not None}
            if len(sc) < 6 or U.get(f"e81_{tag}_d{k}", task) is not None: continue
            if max(sc.values()) <= base:
                print(f"[ed] {tag} draw {k}: guard abstains", flush=True); continue
            todo2.append((tag, key, task, suite, norm, k, sorted(sc, key=sc.get, reverse=True)[:3]))
    print(f"[ed] stage 2: {len(todo2)} commits", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for tag, key, task, suite, norm, k, top in todo2[i:i + U.CHUNK]:
            p = U.build(f"edc_{tag}_{k}", key, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e81_{tag}_d{k}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": suite, "_norm": norm})
        for su in sorted({j["_suite"] for j in jobs}):
            sub = [{a: b for a, b in j.items() if not a.startswith("_")} for j in jobs if j["_suite"] == su]
            U.run(sub, 25, su, next(j["_norm"] for j in jobs if j["_suite"] == su))
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[ed] commit chunk {i//U.CHUNK+1}/{(len(todo2)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[ed] done")

if __name__ == "__main__":
    main()
