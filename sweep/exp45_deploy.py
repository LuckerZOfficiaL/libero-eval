#!/usr/bin/env python3
"""EXP-45 — A: powered deployment test of the learned selector; B: hybrid shortlist + half-budget probe.

36 weak pool-consistent targets with baselines and 8 probed fleet candidates each:
  fold "fleet": the 15 fleet weak targets (base<90%), candidates/probes from e27, probe commit e27_commit_t*_top3
  fold "var":   the 21 weak seed variants (e54), candidates/probes from e54, probe commit e54_<h6>_c
The W+A/48 model is trained on everything EXCEPT the held-out fold (variants otherwise train-only).
Arms, each committed @25 under the same operator:
  model  : model's top-3 of the 8 (features only, no rollouts)
  random : random top-3 of the 8 (seeded)
  hybrid : probe-best top-3 AMONG the model's top-4  (half the probe budget; picks use existing probes)
  probe  : the existing probe top-3 commit (full budget)  [reference]
"""
import argparse, hashlib, json, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import hypernet as H
h6 = lambda name: hashlib.md5(name.encode()).hexdigest()[:6]

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="plan", choices=["plan", "build", "report", "labelspace"])
ap.add_argument("--feats", default="results/analysis/hypernet_featsexta48.npz")
ap.add_argument("--labels", default="results/analysis/pair_labels_ext.json")
ap.add_argument("--seeds", type=int, default=36)
ap.add_argument("--k", type=int, default=4, help="hybrid shortlist size")
ap.add_argument("--blocks", default="WA", help="feature blocks (EXP-85: P = behavioural fingerprint)")
a = ap.parse_args()

plan = json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]
byidx = {x["task_index"]: x for x in plan}
dir_of = {t: x["adapter_file"].split("/")[-2] for t, x in byidx.items()}
# ---- held-out targets and their probe data
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt").read().split("probe rankings: ")[1].strip())
from reinforce_train import scenes as SCENES76, target_baseline
lab54 = json.loads((REPO / "results/analysis/newpool_var50_report.txt").read_text().split("labels: ", 1)[1].strip())
name2idx = {dir_of[t]: t for t in dir_of if t >= 76}
T = {}   # t -> dict(fold, base, probes{p:score}, scenes, suite, probe_commit_job, ext_key)
for t in range(76):
    b = target_baseline(t)
    if b < 0.9 and str(t) in RANK:
        T[t] = dict(fold="fleet", base=b, probes={int(p): s for p, s in RANK[str(t)]}, scenes=SCENES76[t], suite="libero_90",
                    probe_job=f"e27_commit_t{t}_top3", key=t)
for name, v in lab54.items():
    t = name2idx.get(name)
    if t is not None and v["base"] < 0.9:
        T[t] = dict(fold="var", base=v["base"], probes={int(p): s for p, s in v["probes"].items()}, scenes=[name.split("__", 1)[1].split("__seed")[0]],
                    suite=byidx[t]["suite"], probe_job=f"e54_{h6(name)}_c", key="np:" + name)
print(f"targets: {len(T)} (fleet {sum(1 for v in T.values() if v['fold']=='fleet')}, var {sum(1 for v in T.values() if v['fold']=='var')})")

z = np.load(REPO / a.feats); tasks = [int(x) for x in z["tasks"]]; pos = {t: i for i, t in enumerate(tasks)}
F = torch.tensor(z["F"], dtype=torch.float32); sl = json.loads(str(z["slices"]))
F = F[:, np.concatenate([np.arange(*sl[b]) for b in a.blocks])]
L = json.load(open(REPO / a.labels)); groups = {}
for k, v in L.items():
    tt, p = map(int, k.split("_")); groups.setdefault(tt, []).append((p, v["aff"], v["w"]))

def train_and_score(holdout):
    train = [(pos[tt], [pos[p] for p, _, _ in rows], np.array([r[1] for r in rows], np.float32), np.array([r[2] for r in rows], np.float32))
             for tt, rows in groups.items() if tt not in holdout and tt in pos and all(p in pos for p, _, _ in rows) and len(rows) >= 4]
    scores = {t: {} for t in holdout}
    for seed in range(a.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        net = H.AffinityNet(F.shape[1], d_emb=32, p_drop=0.1); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(150):
            net.train()
            for i in np.random.permutation(len(train)):
                tt, cand, aff, lw = train[i]
                s = net(F[tt].expand(len(cand), -1), F[cand]); loss = H.binary_loss(s, aff, lw)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        for t in holdout:
            cand = sorted(T[t]["probes"])
            with torch.no_grad():
                s = net(F[pos[t]].expand(len(cand), -1), F[[pos[c] for c in cand]]).numpy()
            for c, v in zip(cand, s): scores[t][c] = scores[t].get(c, 0.0) + float(v) / a.seeds
    return scores

if a.stage in ("plan", "labelspace"):
    picks = {}
    for fold in ("fleet", "var"):
        hold = sorted(t for t in T if T[t]["fold"] == fold)
        sc = train_and_score(set(hold)); print(f"  fold {fold}: trained excluding {len(hold)} targets", flush=True)
        for t in hold:
            pr = T[t]["probes"]; order = sorted(sc[t], key=sc[t].get, reverse=True)
            rng = np.random.default_rng(int(h6(dir_of[t]), 16) + 11)
            short = order[:a.k]
            picks[t] = dict(model=order[:3], random=sorted(int(x) for x in rng.choice(list(pr), 3, replace=False)),
                            hybrid=sorted(short, key=pr.get, reverse=True)[:3], probe=sorted(pr, key=pr.get, reverse=True)[:3],
                            scores={str(c): round(v, 4) for c, v in sc[t].items()})
    if a.stage == "labelspace":
        # what each policy would have committed, valued by the PROBE affinity of its picks (label space, free)
        from scipy import stats
        val = {arm: [] for arm in ("model", "random", "hybrid", "probe", "oracle")}
        for t, pk in picks.items():
            pr = T[t]["probes"]; b = T[t]["base"]
            for arm in ("model", "random", "hybrid", "probe"): val[arm].append(100 * (np.mean([pr[c] for c in pk[arm]]) - b))
            val["oracle"].append(100 * (np.mean(sorted(pr.values(), reverse=True)[:3]) - b))
        print(f"\n=== EXP-45 label-space: mean probe-affinity of the committed top-3, {len(picks)} weak targets ===")
        for arm in ("random", "model", "hybrid", "probe", "oracle"):
            v = np.array(val[arm]); print(f"  {arm:<8} {v.mean():+6.2f}pp  (SE {stats.sem(v):.2f})" + ("   <- full 8-candidate probe" if arm=="probe" else "") + (f"   <- probe only the model's top-{a.k}: {100*a.k/8:.0f}% of the budget" if arm=="hybrid" else ""))
        d = np.array(val["hybrid"]) - np.array(val["probe"]); print(f"  hybrid - probe: {d.mean():+.2f}pp (SE {stats.sem(d):.2f}, p={stats.ttest_1samp(d,0).pvalue:.3f})")
        d = np.array(val["model"]) - np.array(val["random"]); print(f"  model - random: {d.mean():+.2f}pp (SE {stats.sem(d):.2f}, p={stats.ttest_1samp(d,0).pvalue:.3f})")
    json.dump({str(t): v for t, v in picks.items()}, open(REPO / "results/analysis/exp45_picks.json", "w"), indent=1)
    print("picks written")

if a.stage == "build":
    import unified_pool_driver as U
    picks = {int(k): v for k, v in json.load(open(REPO / "results/analysis/exp45_picks.json")).items()}
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(t, arm) for t in picks for arm in ("model", "random", "hybrid") if not (REPO / f"results/merge/e56_{arm}_{t}.jsonl").exists()]
    print(f"[e45] {len(todo)} commits in chunks of {U.CHUNK}", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built, chunk = [], [], todo[i:i + U.CHUNK]
        for t, arm in chunk:
            key = T[t]["key"]
            if isinstance(key, str): U.EXTPATH[key] = byidx[t]["adapter_file"]
            pth = U.build(f"e45_{arm}_{t}", key, [(c, (ref / U.resid_norm(c)) / 3) for c in picks[t][arm]]); built.append(pth)
            jobs.append({"name": f"e56_{arm}_{t}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [pth],
                         "eval_scenes": T[t]["scenes"], "_suite": T[t]["suite"]})
        for su in sorted({j["_suite"] for j in jobs}):
            sub = [{k: v for k, v in j.items() if k != "_suite"} for j in jobs if j["_suite"] == su]
            U.run(sub, 25, su, U.FLEET_NORM)
        for pth in built: pathlib.Path(pth).unlink(missing_ok=True)
        print(f"[e45] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)
    print("[e45] all commits done")

if a.stage == "report":
    from scipy import stats
    picks = {int(k): v for k, v in json.load(open(REPO / "results/analysis/exp45_picks.json")).items()}
    def rate(job, sc):
        f = REPO / f"results/merge/{job}.jsonl"
        if not f.exists(): return None
        sr = {}
        for ln in open(f):
            j = json.loads(ln)
            if j.get("status") == "ok": sr[j["scene"]] = j["success_rate"]
        ss = [s for s in sc if s in sr]; return sum(sr[s] for s in ss) / len(ss) if len(ss) == len(sc) else None
    rows = []
    for t in sorted(picks):
        b = T[t]["base"]; sc = T[t]["scenes"]
        m, r, h = (rate(f"e56_{arm}_{t}", sc) for arm in ("model", "random", "hybrid")); pr = rate(T[t]["probe_job"], sc)
        if None in (m, r, h): continue
        rows.append((t, T[t]["fold"], b, m, r, h, pr))
    print(f"=== EXP-45: deployed selectors on {len(rows)} weak pool-consistent targets (commit top-3 @25) ===")
    print(f"{'target':<40}{'fold':>6}{'base':>6}{'model':>7}{'random':>8}{'hybrid':>8}{'probe':>7}")
    for t, fo, b, m, r, h, pr in rows:
        print(f"{dir_of[t][:38]:<40}{fo:>6}{100*b:>5.0f}%{100*m:>6.0f}%{100*r:>7.0f}%{100*h:>7.0f}%{('%3.0f%%' % (100*pr)) if pr is not None else '   --':>7}")
    B = np.array([x[2] for x in rows]); M = np.array([x[3] for x in rows]); R = np.array([x[4] for x in rows]); Hh = np.array([x[5] for x in rows])
    P = np.array([x[6] if x[6] is not None else x[2] for x in rows])
    def line(nm, v): print(f"  {nm:<32} {100*(v-B).mean():+6.2f}pp vs base   (SE {100*stats.sem(v-B):.2f})")
    print(); line("random top-3 (unguarded)", R); line("model top-3 (unguarded)", M); line(f"hybrid: probe within model top-{a.k}", Hh); line("probe top-3, full 8 (reference)", P)
    for nm, x, y in (("model - random", M, R), ("hybrid - probe(full)", Hh, P), ("hybrid - random", Hh, R)):
        d = 100 * (x - y); print(f"  {nm:<32} {d.mean():+6.2f}pp  (SE {stats.sem(d):.2f}, p={stats.ttest_1samp(d, 0).pvalue:.3f}, n={len(d)})")
    for fo in ("fleet", "var"):
        mk = np.array([x[1] == fo for x in rows])
        if mk.sum(): print(f"  [{fo} n={int(mk.sum())}] model-random {100*(M[mk]-R[mk]).mean():+.2f}  hybrid-probe {100*(Hh[mk]-P[mk]).mean():+.2f}  probe-base {100*(P[mk]-B[mk]).mean():+.2f}")
