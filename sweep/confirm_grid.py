#!/usr/bin/env python3
"""EXP-60 — PRE-REGISTERED confirmation of dnorm_llm_17 on a new target cohort.

Nested leave-one-target-out on the 15 weak fleet targets picked dnorm_llm_17 on all 15 folds and gave a
held-out interaction correlation of +0.346 (p<0.001), worth 41% of the oracle gap with no rollouts. That
is the best rollout-free result in the project -- and route 1's dnorm_llm_00 looked comparable (31% of
the gap) before a balanced design refuted it. Nested CV controls per-fold selection but not the fact
that this 27-feature family has been explored on these same 15 targets three times.

So this is a confirmation, not a search. STATED IN ADVANCE:
  feature   : dnorm_llm_17 (relative activation-norm change at the final LLM layer), positive sign
  targets   : 10 weak S16-init seed variants -- a different cohort, held out at the DESIGN level
  donors    : the same 8 fixed donors, so the grid decomposition is identical
  prediction: interaction correlation ~ +0.35, and top-3 by the feature worth ~40% of the oracle gap
  failure   : anything near zero refutes it, and the rollout-free feature direction closes for good
No other feature will be promoted from this run; the whole 27-feature table is reported for the record
but the test is dnorm_llm_17 alone.
"""
import hashlib, json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
DES = json.load(open(REPO / "results/analysis/shared_donor_design.json"))
DONORS = DES["donors"]
N_T = 10

def pick_targets():
    lab = json.loads((REPO / "results/analysis/newpool_var50_report.txt").read_text().split("labels: ", 1)[1].strip())
    plan = {x["adapter_file"].split("/")[-2]: x for x in json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]}
    out = []
    for name, v in sorted(lab.items(), key=lambda kv: kv[1]["base"]):
        if 0.30 <= v["base"] <= 0.85 and name in plan:      # real headroom, like the fleet weak cohort
            out.append((name, plan[name]))
        if len(out) == N_T: break
    return out

def main():
    import unified_pool_driver as U
    import train_lora as TL
    tg = pick_targets()
    print(f"[cg] {len(tg)} confirmation targets x {len(DONORS)} donors = {len(tg)*len(DONORS)} cells", flush=True)
    SUITE = {"libero_goal": "libero_goal_no_noops", "libero_object": "libero_object_no_noops",
             "libero_spatial": "libero_spatial_no_noops", "libero_90": "libero_90_no_noops"}
    banks = REPO / "results/analysis/obs_banks"; banks.mkdir(parents=True, exist_ok=True)
    for name, x in tg:                                        # per-target observation banks from own demos
        f = banks / f"{name}.npz"
        if f.exists(): continue
        task = name.split("__", 1)[1].split("__seed")[0]
        s = TL.load_demo_samples(SUITE[x["suite"]], task, n_demos=6, stride=4)
        rng = np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest()[:6], 16))
        idx = rng.choice(len(s), min(60, len(s)), replace=False)
        np.savez_compressed(f, img=np.stack([s[i][0]["observation/image"] for i in idx]),
                            wrist=np.stack([s[i][0]["observation/wrist_image"] for i in idx]),
                            state=np.stack([s[i][0]["observation/state"] for i in idx]),
                            prompt=np.array([x["language"]] * len(idx)), scene=np.array([task] * len(idx)))
        print(f"  bank {name[:48]}: {len(idx)} states", flush=True)
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = []
    for name, x in tg:
        task = name.split("__", 1)[1].split("__seed")[0]
        for d in DONORS:
            if U.get(f"e64_{name[:28]}_p{d}", task) is None:
                todo.append((name, x, task, d))
    print(f"[cg] {len(todo)} probes at 10 rollouts", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for name, x, task, d in todo[i:i + U.CHUNK]:
            key = "np:" + name; U.EXTPATH[key] = x["adapter_file"]
            p = U.build(f"cg_{name[:20]}_{d}", key, [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e64_{name[:28]}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [task], "_suite": x["suite"]})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{k: v for k, v in j.items() if k != "_suite"} for j in jobs if j["_suite"] == su], 10, su, U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[cg] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    json.dump([{"name": n, "task": n.split("__",1)[1].split("__seed")[0], "adapter": x["adapter_file"],
                "suite": x["suite"]} for n, x in tg], open(REPO / "results/analysis/confirm_design.json", "w"), indent=1)
    print("[cg] probes done")

if __name__ == "__main__":
    main()
