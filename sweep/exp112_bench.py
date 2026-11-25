#!/usr/bin/env python3
"""EXP-112 — multi-subset multi-task merging benchmark over fleet + external tasks (user direction 10-09).

Design (fixed before any rollout):
  library  74 fleet LoRAs (LIBERO-90) + 30 external LoRAs (ours, zero-init, LIBERO-Goal/Object/Spatial; shared action
           normalization) = 104 tasks
  subsets  for k in {2, 4, 8}: 3 VALIDATION and 12 TEST subsets, uniform without replacement from the 104 (seeded)
  methods  avg; ta (c/k, c in {0.67, 1.33}; c=1 is avg); ties, dare_ties, tsv, iso_c, iso_cts, knots_ties, ct_merge at
           a NORM-MATCHED scale c * ||avg|| / ||method at scale 1|| (c in {0.67, 1, 1.33}); cart (lambda 0.5, 1);
           ours (pool S16 + alpha * sum r_i, alpha in {1/(2k), 1/k, 2/k}); hook for new candidate methods
  protocol validation: every setting x 3 val subsets, 10 episodes per member task on states 25..34 -> best setting per
           (method, k) by mean member success; test: that setting x 12 test subsets, 10 episodes per member on states
           0..9; single-task baselines: every library task, 2 x 10 episodes on states 0..9 (same period)
  metrics  per test subset: mean retention over members (merged / single-task); win rate of each method vs every other
           over the 36 test subsets (ties = 0.5), average rank, retention split fleet vs external members
Stages (E112_STAGE): base | val | test | report.  Records e112_*.  Env: E112_KS, E112_METHODS, E112_ROUND, E112_DEV,
E112_GPUS, E112_PART (fleet|ext for base).
"""
import hashlib, json, math, os, pathlib, shutil, subprocess, sys, time, zlib
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
import apply_lora as al
import unified_pool_driver as U
import merge_methods as MM
from la_pair_features import load
from reinforce_train import scenes as SC
BASE_DIR = pathlib.Path("/media/pinas/Chris/models/pi05_base_pytorch")
PRE = pathlib.Path("/dev/shm/mergesweep/prebuilt")
DEV = os.environ.get("E112_DEV", "cuda:1")
KS = [int(x) for x in os.environ.get("E112_KS", "2,4,8").split(",")]
ROUND = int(os.environ.get("E112_ROUND", "3"))
NVAL, NTEST = 3, 12
NORMS = REPO / "results/analysis/exp112_norms.json"
h6 = lambda n: hashlib.md5(n.encode()).hexdigest()[:6]

# ------------------------------------------------------------------------------------------------ library + subsets
FLEET = [dict(key=t, tid=f"t{t}", suite="libero_90", scene=SC[t][0], path=str(U.afile[t]), ext=False)
         for t in range(76) if t not in (18, 62)]
EXT = []
for d in sorted((REPO / "results/analysis/new_loras").glob("libero_*_no_noops__*")):
    if (d / "adapter_model.safetensors").exists():
        n = d.name; U.EXTPATH["np:" + n] = str(d / "adapter_model.safetensors")
        EXT.append(dict(key="np:" + n, tid="x" + h6(n), suite=n.split("__")[0].replace("_no_noops", ""),
                        scene=n.split("__", 1)[1], path=str(d / "adapter_model.safetensors"), ext=True))
LIB = FLEET + EXT

def subsets(k, split):
    """stratified: every subset holds n_ext = max(1, round(k * 30/104)) external tasks (k=2,4: 1; k=8: 2), rest fleet"""
    rng = np.random.default_rng(11200 + (100 if split == "test" else 0) + k)
    ne = max(1, round(k * len(EXT) / len(LIB))); fi = list(range(len(FLEET))); ei = list(range(len(FLEET), len(LIB)))
    return [sorted([int(x) for x in rng.choice(fi, k - ne, replace=False)] + [int(x) for x in rng.choice(ei, ne, replace=False)])
            for _ in range(NTEST if split == "test" else NVAL)]

# ------------------------------------------------------------------------------------------------ settings
LINEAR = {"ties": {"topk": 0.2, "lam": 1.0}, "dare_ties": {"p": 0.5, "topk": 0.2, "lam": 1.0}, "tsv": {"alpha": 1.0},
          "iso_c": {"alpha": 1.0}, "iso_cts": {"alpha": 1.0}, "knots_ties": {"topk": 0.2, "lam": 1.0}, "ct_merge": {"gamma": 1.0}}
def settings(k):
    # 10-11 extension (before any test result was read): validation winners sat on grid edges -> extend past them:
    # norm-matched c 1.67, 2.0; ta 2/k; cart 0.25; ours 1/(4k); ours_ties 0.4; ours_rm 3.
    S = [("avg", {}, None)] + [("ta", {"lam": c / k}, None) for c in (0.67, 1.33, 2.0)]
    for m, p in LINEAR.items(): S += [(m, p, c) for c in (0.67, 1.0, 1.33, 1.67, 2.0)]
    S += [("cart", {"lam": v, "gamma": 0.08}, None) for v in (0.25, 0.5, 1.0)]
    S += [("ours", {"alpha": a / k}, None) for a in (0.25, 0.5, 1.0, 2.0)]
    # candidate new methods (10-10): TIES on residuals only; input-weighted residual mean (diagonal RegMean)
    S += [("ours_ties", {"lam": v}, None) for v in (0.4, 0.67, 1.0, 1.5)]
    S += [("ours_rm", {"gamma": v}, None) for v in (0.5, 1.0, 2.0, 3.0)]
    return S

def tag(m, p, c):
    s = m + "".join(f"_{a[0]}{v:.3g}" for a, v in sorted(p.items()) if not (c is not None and a in ("lam", "alpha", "gamma")))
    return s + (f"_c{c:.3g}" if c is not None else "")

# ------------------------------------------------------------------------------------------------ builder
class Builder:
    def __init__(self):
        t0 = time.time(); self.AB = {}
        self.base_sd = st.load_file(str(BASE_DIR / "model.safetensors")); keys = {k: None for k in self.base_sd}
        f0 = load(0); self.AB[0] = f0
        self.ck = {}
        for s_ in U.stems:
            k = al.resolve_base_key(keys, s_)
            if k is not None and s_ in f0: self.ck[s_] = k
        self.norms = json.load(open(NORMS)) if NORMS.exists() else {}
        az = REPO / "results/analysis/exp112_actstats.npz"
        if az.exists():
            z = np.load(az); self.G = z["G"]; self.offs = {k_: tuple(v) for k_, v in json.loads(str(z["offs"])).items()}
            self.stats_done = bool(z["done"].all())
        print(f"[e112] builder ready ({len(self.ck)} modules, {(time.time()-t0)/60:.1f} min)", flush=True)
    def fac(self, i):
        e = LIB[i]
        if e["key"] not in self.AB: self.AB[e["key"]] = load(e["key"]) if not e["ext"] else U.fold(e["key"])
        return self.AB[e["key"]]
    def _delta(self, m, p, members, s_):
        F = [self.fac(i) for i in members]
        if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F): return None
        Bs = [f[s_][0].to(DEV) for f in F]; As = [f[s_][1].to(DEV) for f in F]
        if m == "ours": return MM.ours(Bs, As, p["alpha"], U.USIG[s_].to(DEV), U.VS[s_].to(DEV))
        if m == "ours_ties": return MM.ours_ties(Bs, As, p["lam"], U.USIG[s_].to(DEV), U.VS[s_].to(DEV))
        if m == "ours_rm":
            assert getattr(self, "stats_done", False), "input statistics missing or incomplete"
            o, d = self.offs[s_]
            g = [torch.from_numpy(self.G[i, o:o + d]).to(DEV).clamp(min=1e-12) for i in members]
            return MM.ours_rm(Bs, As, p["gamma"], U.USIG[s_].to(DEV), U.VS[s_].to(DEV), g)
        if m == "dare_ties": return MM.dare_ties(Bs, As, salt=zlib.crc32(s_.encode()) % 100003, **p)
        return getattr(MM, m)(Bs, As, **p)
    def scale(self, m, p, c, members):
        if c is None: return 1.0
        key = f"{m}|{','.join(map(str, members))}"
        if key not in self.norms:
            n1 = na = 0.0
            with torch.no_grad():
                for s_ in self.ck:
                    d = self._delta(m, p, members, s_)
                    if d is None: continue
                    n1 += float(d.norm() ** 2); na += float(self._delta("avg", {}, members, s_).norm() ** 2)
            self.norms[key] = math.sqrt(na / max(n1, 1e-30)); json.dump(self.norms, open(NORMS, "w"))
        return c * self.norms[key]
    def build(self, m, p, c, members, out):
        sc = self.scale(m, p, c, members); sd = dict(self.base_sd)
        with torch.no_grad():
            for s_, ck in self.ck.items():
                D = self._delta(m, p, members, s_)
                if D is None: continue
                W = self.base_sd[ck]; D = sc * D.float(); D = D if D.shape == W.shape else D.T
                sd[ck] = (W.to(DEV).float() + D).to(W.dtype).cpu()
        shutil.rmtree(out, ignore_errors=True); (out / "assets/physical-intelligence/libero").mkdir(parents=True)
        st.save_file(sd, str(out / "model.safetensors"))
        shutil.copy2(BASE_DIR / "config.json", out / "config.json")
        shutil.copy2(REPO / "results/analysis/norm_assets/norm_stats.json", out / "assets/physical-intelligence/libero/norm_stats.json")
        torch.cuda.empty_cache()

# ------------------------------------------------------------------------------------------------ records / jobs
def rec(job, scene):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists(): return None
    for ln in open(f):
        r = json.loads(ln)
        if r.get("status") == "ok" and r["scene"] == scene: return r
    return None
def sjob(tg, k, split, j, suite, part): return f"e112_{tg}_k{k}_{split}{j}_{suite.replace('libero_', '')}-{part}"
def member_jobs(tg, k, split, j, members):
    """one job per (suite, chunk of <= 2 members) -> [(name, suite, [scenes], [member idx])]"""
    out = []
    for su in sorted({LIB[i]["suite"] for i in members}):
        ms = [i for i in members if LIB[i]["suite"] == su]
        for q in range(0, len(ms), 2):
            ch = ms[q:q + 2]; out.append((sjob(tg, k, split, j, su, q // 2), su, [LIB[i]["scene"] for i in ch], ch))
    return out
def plan(jobs, n, offset, tag_):
    meta = dict(U.META)
    gpus = [int(x) for x in os.environ.get("E112_GPUS", "0,1,2").split(",")]
    nw = int(os.environ.get("E112_PER_GPU", "3")) * len(gpus); start = int(os.environ.get("E112_PORT0") or (list(meta.get("port_per_worker") or [8000]))[0])   # E112_PORT0: run 2 orchestrators at once
    meta.update(n_workers=nw, gpu_per_worker=[gpus[i % len(gpus)] for i in range(nw)], port_per_worker=[start + i for i in range(nw)])
    pth = REPO / f"sweep/_e112_{tag_}_{os.getpid()}.json"
    json.dump(dict(meta, n_trials=n, state_offset=offset, jobs=jobs, eval_script="examples/libero/main_dense.py",
                   save_videos="0", heavy_parallel=3), open(pth, "w"), indent=1)
    return pth

def run_items(items, n, offset, split):
    """items: [(m, p, c, k, j, members)] -> rounds of ROUND checkpoints, next round built during evaluation"""
    B = Builder(); rounds = [items[r:r + ROUND] for r in range(0, len(items), ROUND)]
    def build_round(rd):
        js = []
        for m, p, c, k, j, members in rd:
            tg = tag(m, p, c); out = PRE / f"e112_{tg}_k{k}_{split}{j}"; t1 = time.time()
            B.build(m, p, c, members, out)
            for name, su, scenes, ch in member_jobs(tg, k, split, j, members):
                if all(rec(name, s) is not None for s in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [],
                           "prebuilt_dir": str(out), "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e112] built {tg} k={k} {split}{j} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0]) if rounds else []
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                                 str(plan(cur, n, offset, f"{split}{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e112] {split} round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def member_success(tg, k, split, j, members):
    s = {}
    for name, su, scenes, ch in member_jobs(tg, k, split, j, members):
        for i, sc_ in zip(ch, scenes):
            r = rec(name, sc_)
            if r is None:   # 11-06: member order (hence chunk) can differ across processes; every chunk of a tag serves the same merge
                r = next((x for q in range(len(members)) if (x := rec(sjob(tg, k, split, j, su, q), sc_)) is not None), None)
            if r is not None: s[i] = r["success_rate"]
    return s

def best_settings(methods):
    out = {}
    for k in KS:
        for m in methods:
            cand = [(mm, p, c) for mm, p, c in settings(k) if mm == m]; sc = {}
            for mm, p, c in cand:
                v = [np.mean(list(member_success(tag(mm, p, c), k, "val", j, S).values()) or [np.nan]) for j, S in enumerate(subsets(k, "val"))]
                sc[tag(mm, p, c)] = (np.nan if any(np.isnan(v)) else float(np.mean(v)), (mm, p, c))
            ok = {t_: v for t_, v in sc.items() if not np.isnan(v[0])}
            if ok:
                bt = max(ok, key=lambda t_: ok[t_][0]); out[(m, k)] = ok[bt][1]
                print(f"[e112] k={k} {m}: best {bt} (val {ok[bt][0]:.3f}; " + ", ".join(f"{t_} {v[0]:.3f}" for t_, v in sc.items()) + ")", flush=True)
    return out

def main():
    stage = os.environ.get("E112_STAGE", "base"); U._clear_orphaned_builds()
    methods = os.environ["E112_METHODS"].split(",") if os.environ.get("E112_METHODS") else \
        ["avg", "ta", "ties", "dare_ties", "tsv", "iso_c", "iso_cts", "knots_ties", "cart", "ct_merge", "ours", "ours_ties", "ours_rm"]
    if stage == "base":
        part = os.environ.get("E112_PART", "all")
        lib = [e for e in LIB if part == "all" or (part == "ext") == e["ext"]]
        J = [{"name": f"e112_base_{e['tid']}_r{r}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [e["path"]],
              "eval_scenes": [e["scene"]], "suite": e["suite"], "norm_stats": U.FLEET_NORM}
             for r in (1, 2) for e in lib if rec(f"e112_base_{e['tid']}_r{r}", e["scene"]) is None]
        print(f"[e112] base ({part}): {len(J)} single-task jobs x 10", flush=True)
        if J: subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(plan(J, 10, 0, "base"))], cwd=REPO)
    elif stage == "val":
        items = [(m, p, c, k, j, S) for k in KS for m, p, c in settings(k) if m in methods for j, S in enumerate(subsets(k, "val"))
                 if any(rec(nm, s) is None for nm, su, scs, ch in member_jobs(tag(m, p, c), k, "val", j, S) for s in scs)]
        print(f"[e112] val: {len(items)} merged models", flush=True); run_items(items, 10, 25, "val")
    elif stage == "test":
        best = best_settings(methods)
        full = [(m, p, c, k, j, S) for (mm, k), (m, p, c) in sorted(best.items(), key=lambda x: (x[0][1], x[0][0]))
                for j, S in enumerate(subsets(k, "test"))]
        spec = os.environ.get("E112_SHARD")                                    # e.g. "0,1,2,3/5"; else per-host file
        sf = REPO / "results/analysis/exp112_shard.json"
        if not spec and sf.exists():
            import socket; spec = json.load(open(sf)).get(socket.gethostname())
        if spec:
            keep, n_ = spec.split("/"); keep = {int(x) for x in keep.split(",")}
            full = [it for q, it in enumerate(full) if q % int(n_) in keep]
            print(f"[e112] test shard {spec}: {len(full)} of the full list", flush=True)
        items = [it for it in full
                 if any(rec(nm, s) is None for nm, su, scs, ch in member_jobs(tag(it[0], it[1], it[2]), it[3], "test", it[4], it[5]) for s in scs)]
        print(f"[e112] test: {len(items)} merged models", flush=True); run_items(items, 10, 0, "test")
    else:
        report(methods)
    print(f"[e112] {stage} DONE", flush=True)

def report(methods):
    single = {}
    for i, e in enumerate(LIB):
        rs = [rec(f"e112_base_{e['tid']}_r{r}", e["scene"]) for r in (1, 2)]; rs = [x for x in rs if x]
        if rs: single[i] = np.mean([x["success_rate"] for x in rs])
    best = best_settings(methods); L = ["=== EXP-112 multi-subset benchmark (test subsets, 10 eps per member on states 0..9) ==="]
    R = {}                                                                          # (m, k, j) -> (ret_all, ret_fleet, ret_ext)
    for (m, k), (mm, p, c) in best.items():
        for j, S in enumerate(subsets(k, "test")):
            s = member_success(tag(mm, p, c), k, "test", j, S)
            if len(s) < len(S): continue
            ret = {i: s[i] / single[i] for i in S if i in single and single[i] > 0}
            if not ret: continue
            R[(m, k, j)] = (np.mean(list(ret.values())), np.mean([v for i, v in ret.items() if not LIB[i]["ext"]] or [np.nan]),
                            np.mean([v for i, v in ret.items() if LIB[i]["ext"]] or [np.nan]))
    for k in KS:
        ms = [m for m in methods if all((m, k, j) in R for j in range(NTEST))]
        if not ms: L.append(f"k={k}: incomplete"); continue
        L.append(f"\nk={k}: {len(ms)} methods x {NTEST} test subsets")
        ranks = {m: [] for m in ms}
        for j in range(NTEST):
            v = sorted(ms, key=lambda m: -R[(m, k, j)][0])
            for r_, m in enumerate(v): ranks[m].append(r_ + 1)
        for m in sorted(ms, key=lambda m: np.mean(ranks[m])):
            a = np.array([R[(m, k, j)] for j in range(NTEST)])
            L.append(f"  {m:11s} retention {100*np.nanmean(a[:,0]):5.1f}% (fleet {100*np.nanmean(a[:,1]):5.1f}%, ext {100*np.nanmean(a[:,2]):5.1f}%)  avg rank {np.mean(ranks[m]):.2f}")
        for o_ in [m for m in ms if m.startswith("ours")]:
            wins = {b: np.mean([1.0 if R[(o_, k, j)][0] > R[(b, k, j)][0] else 0.5 if R[(o_, k, j)][0] == R[(b, k, j)][0] else 0.0
                                for j in range(NTEST)]) for b in ms if not b.startswith("ours")}
            L.append(f"  {o_} win rate vs: " + ", ".join(f"{b} {100*w:.0f}%" for b, w in sorted(wins.items(), key=lambda x: -x[1])))
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp112_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
