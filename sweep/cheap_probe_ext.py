#!/usr/bin/env python3
"""EXP-64 — resolve the cheap probe at larger n, using targets whose standard arm already exists.

The deployed comparison (EXP-51) put 8x5 at 69% of 8x10's gain with a paired difference of -2.67pp,
p=0.32 at n=15 -- ambiguous, and EXP-63 has since shown per-target noise is ~+/-20pp, so n=15 was never
going to settle it. The 21 weak S16-init variants already carry a full 8x10 probe and a committed merge
from EXP-42, so only the CHEAP arm needs rollouts: 8 candidates x 5 rollouts, its own top-3, commit @25.
That takes the comparison from n=15 to n=36 for ~1,365 rollouts.
"""
import hashlib, json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
h6 = lambda n: hashlib.md5(n.encode()).hexdigest()[:6]
lab = json.loads((REPO / "results/analysis/newpool_var50_report.txt").read_text().split("labels: ", 1)[1].strip())
plan = {x["adapter_file"].split("/")[-2]: x for x in json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]}
TG = [(n, v) for n, v in sorted(lab.items()) if v["base"] < 0.9 and n in plan]

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    for n, v in TG:
        U.EXTPATH["np:" + n] = plan[n]["adapter_file"]
    print(f"[cx] {len(TG)} weak variant targets; standard arm already measured (e54)", flush=True)
    todo = [(n, v, int(p)) for n, v in TG for p in v["probes"]
            if U.get(f"e68_{h6(n)}_p{p}", n.split('__',1)[1].split('__seed')[0]) is None]
    print(f"[cx] stage 1: {len(todo)} probes at 5 rollouts", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for n, v, d in todo[i:i + U.CHUNK]:
            task = n.split('__',1)[1].split('__seed')[0]
            p = U.build(f"cx_{h6(n)}_{d}", "np:" + n, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e68_{h6(n)}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": plan[n]["suite"]})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{k: x for k, x in j.items() if k != "_suite"} for j in jobs if j["_suite"] == su], 5, su, U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[cx] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    todo2 = []
    for n, v in TG:
        task = n.split('__',1)[1].split('__seed')[0]
        sc = {int(p): U.get(f"e68_{h6(n)}_p{p}", task) for p in v["probes"]}
        sc = {k: x for k, x in sc.items() if x is not None}
        if len(sc) < 4 or U.get(f"e69_{h6(n)}_c", task) is not None: continue
        if max(sc.values()) <= v["base"]:
            print(f"[cx] {n[:40]}: cheap probe abstains", flush=True); continue
        todo2.append((n, v, task, sorted(sc, key=sc.get, reverse=True)[:3]))
    print(f"[cx] stage 2: {len(todo2)} commits at 25 rollouts", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for n, v, task, top in todo2[i:i + U.CHUNK]:
            p = U.build(f"cxc_{h6(n)}", "np:" + n, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e69_{h6(n)}_c", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": plan[n]["suite"]})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{k: x for k, x in j.items() if k != "_suite"} for j in jobs if j["_suite"] == su], 25, su, U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[cx] commit chunk {i//U.CHUNK+1}/{(len(todo2)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[cx] done")

if __name__ == "__main__":
    main()
