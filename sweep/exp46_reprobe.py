#!/usr/bin/env python3
"""EXP-46/47 — re-probe the SAME (target, candidate) pairs a second time.

Two questions, one run:
 (46) Is the behavioural probe less reliable on freshly trained targets than on settled fleet ones?
      Test-retest Spearman per target between run 1 (e27 / e54) and this run. Fleet reference: 0.41
      (EXP-22.4, same-adapter repeats). A matched-baseline analysis showed the fresh-target harm rate
      is mostly a headroom artifact, so this measures the mechanism directly instead of inferring it.
 (47) How cheap can the probe get? Two independent 10-rollout runs per pair give real repeated
      measurements: pooling them gives a 20-rollout probe, each alone gives 10, and splitting gives 5.
      So selection quality vs budget can be estimated without assuming subsample exchangeability.

Targets: 5 settled fleet weak targets (candidates from e27) + 5 fresh S16 1.5k variants with matched
baselines (candidates from e54). Probe only -- baselines and commits already exist.
"""
import hashlib, json, pathlib, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC76, target_baseline
h6 = lambda n: hashlib.md5(n.encode()).hexdigest()[:6]

FLEET = [6, 19, 50, 15, 56]
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
lab = json.loads((REPO / "results/analysis/newpool_var50_report.txt").read_text().split("labels: ", 1)[1].strip())
plan = json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]
idx_of = {x["adapter_file"].split("/")[-2]: x for x in plan}
# fresh variants: baseline inside the settled cohort's band, one per distinct base task
fresh, seen = [], set()
for name, v in sorted(lab.items(), key=lambda kv: -kv[1]["base"]):
    task = name.split("__", 1)[1].split("__seed")[0]
    if 0.55 <= v["base"] <= 0.80 and task not in seen and name in idx_of:
        fresh.append(name); seen.add(task)
    if len(fresh) == 5: break

TG = []   # (tag, key, task-list, suite, {cand: run1_rate}, base)
for t in FLEET:
    TG.append((f"f{t}", t, SC76[t], "libero_90", {int(p): s for p, s in RANK[str(t)]}, target_baseline(t)))
for name in fresh:
    x = idx_of[name]
    TG.append((f"v{h6(name)}", "np:" + name, [name.split("__", 1)[1].split("__seed")[0]], x["suite"],
               {int(p): s for p, s in lab[name]["probes"].items()}, lab[name]["base"]))
    U.EXTPATH["np:" + name] = x["adapter_file"]

if __name__ == "__main__":
    U._clear_orphaned_builds()
    print(f"[e46] {len(TG)} targets: {[t[0] for t in TG]}", flush=True)
    for tag, key, sc, suite, pr, base in TG:
        print(f"  {tag:<10} base {100*base:3.0f}%  {len(pr)} candidates  suite {suite}", flush=True)
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(tag, key, sc, suite, c) for tag, key, sc, suite, pr, _ in TG for c in sorted(pr)
            if not (REPO / f"results/merge/e57_{tag}_p{c}.jsonl").exists()]
    print(f"[e46] {len(todo)} re-probes in chunks of {U.CHUNK}", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built, chunk = [], [], todo[i:i + U.CHUNK]
        for tag, key, sc, suite, c in chunk:
            p = U.build(f"e46_{tag}_{c}", key, [(c, ref / U.resid_norm(c))]); built.append(p)
            jobs.append({"name": f"e57_{tag}_p{c}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": sc, "_suite": suite})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{k: v for k, v in j.items() if k != "_suite"} for j in jobs if j["_suite"] == su], 10, su, U.FLEET_NORM)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
        print(f"[e46] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)
    print("[e46] done")
