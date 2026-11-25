#!/usr/bin/env python3
"""EXP-76 — extend draw-averaging to the S16-init variant cohort (odin).

EXP-73 showed evaluation noise is not the binding constraint: doubling episodes per commit did not
tighten the headline, because candidate-draw variance (13.6pp) dominates. Draw-averaging does tighten it
(EXP-74: SE 2.32 -> 1.83). The 20 weak S16-init variants were measured with ONE draw each in EXP-42, so
a second independent draw per target extends the draw-averaged estimate from 15 targets to 35.
Given to odin because it is long, undemanding evaluation work; shiva stays free for the external claim.
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

def cands(name):
    rng = np.random.default_rng(int(h6(name + "_draw2"), 16))
    return sorted(int(x) for x in rng.choice(76, 8, replace=False))

def main():
    U._clear_orphaned_builds()
    for n, v in TG:
        U.EXTPATH["np:" + n] = plan[n]["adapter_file"]
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    print(f"[vd] {len(TG)} weak variant targets, one further independent draw each", flush=True)
    todo = [(n, v, d) for n, v in TG for d in cands(n)
            if U.get(f"e82_{h6(n)}_p{d}", n.split('__',1)[1].split('__seed')[0]) is None]
    print(f"[vd] stage 1: {len(todo)} probes", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for n, v, d in todo[i:i + U.CHUNK]:
            task = n.split('__',1)[1].split('__seed')[0]
            p = U.build(f"vd_{h6(n)}_{d}", "np:" + n, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e82_{h6(n)}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": plan[n]["suite"]})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{a: b for a, b in j.items() if a != "_suite"} for j in jobs if j["_suite"] == su], 10, su, U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[vd] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    todo2 = []
    for n, v in TG:
        task = n.split('__',1)[1].split('__seed')[0]
        sc = {d: U.get(f"e82_{h6(n)}_p{d}", task) for d in cands(n)}
        sc = {a: x for a, x in sc.items() if x is not None}
        if len(sc) < 6 or U.get(f"e83_{h6(n)}_c", task) is not None: continue
        if max(sc.values()) <= v["base"]:
            print(f"[vd] {n[:38]}: guard abstains", flush=True); continue
        todo2.append((n, task, plan[n]["suite"], sorted(sc, key=sc.get, reverse=True)[:3]))
    print(f"[vd] stage 2: {len(todo2)} commits", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for n, task, suite, top in todo2[i:i + U.CHUNK]:
            p = U.build(f"vdc_{h6(n)}", "np:" + n, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e83_{h6(n)}_c", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": suite})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{a: b for a, b in j.items() if a != "_suite"} for j in jobs if j["_suite"] == su], 25, su, U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[vd] commit chunk {i//U.CHUNK+1}/{(len(todo2)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[vd] done")

if __name__ == "__main__":
    main()
