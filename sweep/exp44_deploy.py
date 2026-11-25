#!/usr/bin/env python3
"""EXP-44 — deploy the learned selector on 16 unseen pool-consistent targets and MERGE its picks.

Label-space gains are not the goal; selection at merge time is. The 16 targets (6 EXP-41 short adapters
+ 10 EXP-38 seed variants) have baselines and 8 probed fleet candidates each, and the probe's top-3
commits already exist (e52_*_c / e53_*_c). Here the W+A/48 affinity model, trained on everything
EXCEPT these 16, scores the same 8 candidates from features alone (no rollouts) and we commit:
    model top-3 @25   vs   random top-3 @25 (seeded)   vs   probe top-3 (existing)
Writes the merge plan for shiva (build() from unified_pool_driver -> bit-identical operator).
"""
import argparse, hashlib, json, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep"))
import hypernet as H

h6 = lambda name: hashlib.md5(name.encode()).hexdigest()[:6]
ap = argparse.ArgumentParser()
ap.add_argument("--feats", default="results/analysis/hypernet_featsexta48.npz")
ap.add_argument("--labels", default="results/analysis/pair_labels_ext.json")
ap.add_argument("--baselines", default="results/analysis/baselines_ext.json")
ap.add_argument("--seeds", type=int, default=36)
ap.add_argument("--stage", default="plan", choices=["plan", "build", "report"])
a = ap.parse_args()

plan = json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]
dir_of = {x["task_index"]: x["adapter_file"].split("/")[-2] for x in plan}
HOLD = sorted(x["task_index"] for x in plan if x["adapter_file"].split("/")[-3] in ("new_loras_short", "new_loras_seeds"))
assert len(HOLD) == 16, HOLD
z = np.load(REPO / a.feats); tasks = [int(x) for x in z["tasks"]]; pos = {t: i for i, t in enumerate(tasks)}
F = torch.tensor(z["F"], dtype=torch.float32)
sl = json.loads(str(z["slices"])); keep = np.concatenate([np.arange(*sl[b]) for b in "WA"]); F = F[:, keep]
L = json.load(open(REPO / a.labels))
groups = {}
for k, v in L.items():
    t, p = map(int, k.split("_")); groups.setdefault(t, []).append((p, v["aff"], v["w"]))
train = [(pos[t], [pos[p] for p, _, _ in rows], np.array([r[1] for r in rows], np.float32), np.array([r[2] for r in rows], np.float32))
         for t, rows in groups.items() if t not in HOLD and t in pos and all(p in pos for p, _, _ in rows) and len(rows) >= 4]
print(f"train targets {len(train)} (excluding the 16 held-out); features {tuple(F.shape)}")

if a.stage == "plan":
    scores = {t: {} for t in HOLD}
    for seed in range(a.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        net = H.AffinityNet(F.shape[1], d_emb=32, p_drop=0.1)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(150):
            net.train()
            for i in np.random.permutation(len(train)):
                T, cand, aff, lw = train[i]
                s = net(F[T].expand(len(cand), -1), F[cand]); loss = H.binary_loss(s, aff, lw)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        for t in HOLD:
            cand = [p for p, _, _ in groups[t]]
            with torch.no_grad():
                s = net(F[pos[t]].expand(len(cand), -1), F[[pos[c] for c in cand]]).numpy()
            for c, v in zip(cand, s): scores[t][c] = scores[t].get(c, 0.0) + float(v) / a.seeds
        if seed % 6 == 5: print(f"  seed {seed+1}/{a.seeds}", flush=True)
    picks = {}
    for t in HOLD:
        cand = sorted(scores[t], key=scores[t].get, reverse=True)
        rng = np.random.default_rng(int(h6(dir_of[t]), 16) + 7)
        picks[dir_of[t]] = {"model": cand[:3], "random": sorted(int(x) for x in rng.choice(list(scores[t]), 3, replace=False)),
                            "scores": {str(c): round(v, 4) for c, v in scores[t].items()}, "base_task": dir_of[t].split("__", 1)[1].split("__seed")[0]}
    json.dump(picks, open(REPO / "results/analysis/exp44_picks.json", "w"), indent=1)
    # agreement with the probe's ranking as a free sanity metric
    from scipy import stats
    rho = [stats.spearmanr([scores[t][p] for p, _, _ in groups[t]], [af for _, af, _ in groups[t]]).correlation for t in HOLD]
    print(f"picks written; model-vs-probe Spearman on the 16 targets: mean {np.nanmean(rho):+.2f}")


if a.stage == "build":
    # build -> eval -> delete in CHUNKS (all 32 at once is ~18GB and blew shiva's quota on 09-07)
    import unified_pool_driver as U
    picks = json.load(open(REPO / "results/analysis/exp44_picks.json"))
    path_of = {x["adapter_file"].split("/")[-2]: x for x in plan}
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = []
    for d, pk in picks.items():
        for arm in ("model", "random"):
            name = f"e55_{arm}_{h6(d)}"
            if not (REPO / f"results/merge/{name}.jsonl").exists():
                todo.append((d, arm, name))
    print(f"[e44] {len(todo)} commits to run in chunks of {U.CHUNK}", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for d, arm, name in todo[i:i + U.CHUNK]:
            key = "np:" + d; U.EXTPATH[key] = path_of[d]["adapter_file"]
            pk = picks[d]
            pth = U.build(f"e44_{arm}_{h6(d)}", key, [(c, (ref / U.resid_norm(c)) / 3) for c in pk[arm]])
            built.append(pth)
            jobs.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [pth],
                         "eval_scenes": [pk["base_task"]]})
        U.run(jobs, 25, path_of[todo[i][0]]["suite"], U.FLEET_NORM) if len({path_of[d]["suite"] for d, _, _ in todo[i:i+U.CHUNK]}) == 1 else None
        # mixed suites within a chunk: run per suite
        if len({path_of[d]["suite"] for d, _, _ in todo[i:i+U.CHUNK]}) > 1:
            for su in sorted({path_of[d]["suite"] for d, _, _ in todo[i:i+U.CHUNK]}):
                sub = [j for j, (d, _, _) in zip(jobs, todo[i:i+U.CHUNK]) if path_of[d]["suite"] == su]
                U.run(sub, 25, su, U.FLEET_NORM)
        for pth in built:
            pathlib.Path(pth).unlink(missing_ok=True)
        print(f"[e44] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)
    print("[e44] all commits done")

if a.stage == "report":
    from scipy import stats
    picks = json.load(open(REPO / "results/analysis/exp44_picks.json"))
    B = {int(k): v for k, v in json.load(open(REPO / a.baselines)).items()}
    def rate(job):
        f = REPO / f"results/merge/{job}.jsonl"
        if not f.exists(): return None
        r = None
        for ln in open(f):
            j = json.loads(ln)
            if j.get("status") == "ok": r = j["success_rate"]
        return r
    rows = []
    for d, pk in picks.items():
        t = next(x["task_index"] for x in plan if x["adapter_file"].split("/")[-2] == d)
        tagp = ("e52" if "new_loras_short" in [x["adapter_file"].split("/")[-3] for x in plan if x["task_index"] == t] else "e53") + f"_{h6(d)}"
        base = B.get(t); m = rate(f"e55_model_{h6(d)}"); r = rate(f"e55_random_{h6(d)}"); pr = rate(f"{tagp}_c")
        if base is None or m is None or r is None: continue
        rows.append((d, base, m, r, pr))
    print(f"=== EXP-44: learned selector DEPLOYED on {len(rows)} unseen pool-consistent targets (commit top-3 @25) ===")
    print(f"{'target':<52}{'base':>6}{'model':>7}{'random':>8}{'probe':>7}")
    for d, b, m, r, pr in sorted(rows, key=lambda x: x[1]):
        print(f"{d[:50]:<52}{100*b:>5.0f}%{100*m:>6.0f}%{100*r:>7.0f}%{('%3.0f%%' % (100*pr)) if pr is not None else '  abst':>7}")
    if rows:
        b = np.array([x[1] for x in rows]); m = np.array([x[2] for x in rows]); r = np.array([x[3] for x in rows])
        pr = np.array([x[4] if x[4] is not None else x[1] for x in rows])
        dm = 100*(m-r); print(f"\n  model - random: {dm.mean():+.2f}pp (SE {stats.sem(dm):.2f}, p={stats.ttest_1samp(dm,0).pvalue:.3f});  "
              f"model vs base {100*(m-b).mean():+.2f}; random vs base {100*(r-b).mean():+.2f}; probe(guarded) vs base {100*(pr-b).mean():+.2f}")
        w = b < 0.9
        if w.sum(): print(f"  WEAK n={int(w.sum())}: model {100*(m[w]-b[w]).mean():+.2f}  random {100*(r[w]-b[w]).mean():+.2f}  probe {100*(pr[w]-b[w]).mean():+.2f}")
