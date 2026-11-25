#!/usr/bin/env python3
"""EXP-77 — draw-average the manufactured-consistency claim (S16-init targets with headroom).

The S16 story ends with: new experts initialised from S16 are genuine pool members, and the method then
works on them (+11.0pp, n=4, ONE draw). That is the thinnest of our three headline claims and the only
one still single-draw, after EXP-74 (fleet, +7.98) and EXP-75 (external, +5.33) were averaged. Two
further independent per-target draws for every S16-init target that has real headroom.
"""
import hashlib, json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
h6 = lambda n: hashlib.md5(n.encode()).hexdigest()[:6]
SUITE = {"libero_goal": "libero_goal", "libero_object": "libero_object",
         "libero_spatial": "libero_spatial", "libero_90": "libero_90"}
plan = {x["adapter_file"].split("/")[-2]: x for x in json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]}
TG = []
for rep in ("newpool_short_report.txt", "veryweak_report.txt"):
    f = REPO / "results/analysis" / rep
    if not f.exists(): continue
    for n, v in json.loads(f.read_text().split("labels: ", 1)[1].strip()).items():
        if v["base"] < 0.9:
            d = next((p for p in (REPO/"results/analysis/new_loras_short"/n, REPO/"results/analysis/new_loras_weak"/n)
                      if (p/"adapter_model.safetensors").exists()), None)
            if d: TG.append((n, v["base"], str(d/"adapter_model.safetensors"),
                             n.split("__")[0].replace("_no_noops", ""), n.split("__",1)[1].split("__seed")[0]))
DRAWS = [1, 2]

def cands(name, k):
    rng = np.random.default_rng(int(h6(f"{name}_s16draw{k}"), 16))
    return sorted(int(x) for x in rng.choice(76, 8, replace=False))

def main():
    U._clear_orphaned_builds()
    for n, b, path, suite, task in TG: U.EXTPATH["np:" + n] = path
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    print(f"[sd2] {len(TG)} S16-init targets with headroom x {len(DRAWS)} further draws", flush=True)
    todo = [(g, k, d) for g in TG for k in DRAWS for d in cands(g[0], k)
            if U.get(f"e84_{h6(g[0])}_d{k}_p{d}", g[4]) is None]
    print(f"[sd2] stage 1: {len(todo)} probes", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for (n, b, path, suite, task), k, d in todo[i:i + U.CHUNK]:
            p = U.build(f"sd2_{h6(n)}_{k}_{d}", "np:" + n, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e84_{h6(n)}_d{k}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": suite})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{a: x for a, x in j.items() if a != "_suite"} for j in jobs if j["_suite"] == su], 10, su, U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[sd2] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    todo2 = []
    for n, b, path, suite, task in TG:
        for k in DRAWS:
            sc = {d: U.get(f"e84_{h6(n)}_d{k}_p{d}", task) for d in cands(n, k)}
            sc = {a: x for a, x in sc.items() if x is not None}
            if len(sc) < 6 or U.get(f"e85_{h6(n)}_d{k}", task) is not None: continue
            if max(sc.values()) <= b:
                print(f"[sd2] {n[:36]} draw {k}: guard abstains", flush=True); continue
            todo2.append((n, suite, task, k, sorted(sc, key=sc.get, reverse=True)[:3]))
    print(f"[sd2] stage 2: {len(todo2)} commits", flush=True)
    for i in range(0, len(todo2), U.CHUNK):
        jobs, built = [], []
        for n, suite, task, k, top in todo2[i:i + U.CHUNK]:
            p = U.build(f"sd2c_{h6(n)}_{k}", "np:" + n, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e85_{h6(n)}_d{k}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": suite})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{a: x for a, x in j.items() if a != "_suite"} for j in jobs if j["_suite"] == su], 25, su, U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[sd2] commit chunk {i//U.CHUNK+1}/{(len(todo2)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[sd2] done")

if __name__ == "__main__":
    main()
