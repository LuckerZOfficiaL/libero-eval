#!/usr/bin/env python3
"""EXP-110 — multi-task merging benchmark: our operator vs established methods, tuned on held-out states.

Design (fixed before any rollout; user-approved 10-07):
  groups   EXP-109's seeded partition of the 74 unique LIBERO-90 tasks into groups of k (primary k = 16: 5 groups
           contain the 24-task panel); one merged model per (setting, group), scored on the group's panel tasks.
  methods  avg, ta, ties, dare_ties, tsv, iso_c, iso_cts, knots_ties, cart, ct_merge (baselines; sweep/merge_methods.py)
           ours (pool S16 + alpha * sum of two-sided residuals) + ablations ours_pgc (pool subspace, group
           coefficients), ours_group (group-only subspace), ours_lgo (leave-group-out pool subspace)
  stage val   every grid setting: 10 episodes per panel task on initial states 25..34 -> best setting per method
              (pooled success over the 24 panel tasks; ties -> the earlier grid value)
  stage test  each method's best setting: 25 episodes per panel task on states 0..24 (never used for any choice)
  metric   retention = merged success / single-task success (EXP-109 e109_base_t{t}, same panel, same states)
Dense merged checkpoints are built on GPU module by module and served as prebuilt dirs (rounds; next round is built
while the current one is evaluated). Records e110_{setting}_k{k}_g{i}_{val|test}.
Env: E110_STAGE val|test|report, E110_K (16), E110_METHODS (subset), E110_ROUND (4), E110_DEV (cuda:2).
"""
import json, math, os, pathlib, shutil, subprocess, sys, time, zlib
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
import apply_lora as al
import unified_pool_driver as U
import merge_methods as MM
from la_pair_features import load
from reinforce_train import scenes as SC
from exp109_consolidate import PANEL, panel_groups, rec
BASE_DIR = pathlib.Path("/media/pinas/Chris/models/pi05_base_pytorch")
PRE = pathlib.Path("/dev/shm/mergesweep/prebuilt")
K = int(os.environ.get("E110_K", "16")); ROUND = int(os.environ.get("E110_ROUND", "4"))      # checkpoints per round (7 GB each; next round built during eval)
DEV = os.environ.get("E110_DEV", "cuda:2")
DUP = {7: 18, 18: 7, 29: 62, 62: 29}

def grid(k):
    # 10-08 extension (before any test run): the first grids came from vision benchmarks; measured update norms at
    # scale 1 relative to averaging (k=16): ta 16x, tsv 4.5x, iso_c 3.3x, iso_cts 3.4x, ct 2.1x, dare 2.6x, ties 1.4x,
    # knots 1.3x, cart 1.0x. Added size-matched values (~0.67x / 1x / 1.33x of averaging) and values past every grid edge
    # that won. ta at 1/k is identical to avg (not re-run).
    q = 1 / math.sqrt(k)
    return {"avg": [{}],
            "ta": [{"lam": v} for v in (0.04, 0.08, 0.1, 0.2, 0.3, 0.5)],
            "ties": [{"topk": 0.2, "lam": v} for v in (0.5, 1.0, 1.5)],
            "dare_ties": [{"p": 0.5, "topk": 0.2, "lam": v} for v in (0.35, 0.5, 1.0, 1.5)],
            "tsv": [{"alpha": v} for v in (0.15, 0.22, 0.3, 0.5, 1.0, 1.5)],
            "iso_c": [{"alpha": v} for v in (0.2, 0.31, 0.41, 0.5, 1.0, 1.5)],
            "iso_cts": [{"alpha": v} for v in (0.19, 0.29, 0.39, 0.75, 1.25, 1.75)],
            "knots_ties": [{"topk": 0.2, "lam": v} for v in (0.3, 0.6, 1.0, 1.25, 1.5)],
            "cart": [{"lam": v, "gamma": 0.08} for v in (0.5, 1.0, 1.5, 2.0)],
            "ct_merge": [{"gamma": v} for v in (0.15, 0.3, 0.45, 0.6, 1.0)],
            "ours": [{"alpha": v} for v in (0.0, 1 / (2 * k), 1 / k, q, 0.5, 1.0)],
            "ours_pgc": [{"alpha": v} for v in (0.125, q, 0.5, 1.0)],
            "ours_group": [{"alpha": v} for v in (0.5, 1.0)],
            "ours_lgo": [{"alpha": v} for v in (1 / k, 0.125, q, 0.5, 1.0)]}

# the 41 settings of the first validation round (done before the 10-08 extension)
OLD_TAGS = {"avg"} | {f"ta_l{v:.3g}" for v in (0.1, 0.2, 0.3, 0.5)} | {f"ties_l{v:.3g}_t0.2" for v in (0.5, 1.0, 1.5)} \
    | {f"dare_ties_l{v:.3g}_p0.5_t0.2" for v in (0.5, 1.0, 1.5)} | {f"tsv_a{v:.3g}" for v in (0.5, 1.0, 1.5)} \
    | {f"iso_c_a{v:.3g}" for v in (0.5, 1.0, 1.5)} | {f"iso_cts_a{v:.3g}" for v in (0.75, 1.25, 1.75)} \
    | {f"knots_ties_l{v:.3g}_t0.2" for v in (0.3, 0.6, 1.0)} | {f"cart_g0.08_l{v:.3g}" for v in (1.0, 1.5, 2.0)} \
    | {f"ct_merge_g{v:.3g}" for v in (0.3, 0.6, 1.0)} | {f"ours_a{v:.3g}" for v in (1 / 16, 0.25, 0.5, 1.0)} \
    | {f"ours_pgc_a{v:.3g}" for v in (0.25, 0.5, 1.0)} | {f"ours_group_a{v:.3g}" for v in (0.5, 1.0)} \
    | {f"ours_lgo_a{v:.3g}" for v in (0.25, 0.5, 1.0)}

def tag(m, p): return m + "".join(f"_{k[0]}{v:.3g}" for k, v in sorted(p.items()))
def job(m, p, k, i, stage): return f"e110_{tag(m, p)}_k{k}_g{i}_{stage}"
def sjob(m, p, k, i, stage, t): return f"{job(m, p, k, i, stage)}_t{t}"          # one job per panel task

class Builder:
    def __init__(self):
        t0 = time.time()
        self.AB = [load(t) for t in range(76)]
        self.base_sd = st.load_file(str(BASE_DIR / "model.safetensors"))
        keys = {k: None for k in self.base_sd}
        self.ck = {}
        for s_ in U.stems:
            k = al.resolve_base_key(keys, s_)
            if k is None or not all(s_ in self.AB[t] for t in range(76)): continue
            if all(float(self.AB[t][s_][0].abs().max()) == 0.0 for t in range(76)): continue      # untrained modules
            self.ck[s_] = k
        print(f"[e110] builder: {len(self.ck)} trained modules ({(time.time()-t0)/60:.1f} min)", flush=True)
    def delta(self, m, p, G, s_):
        Bs, As = [self.AB[t][s_][0].to(DEV) for t in G], [self.AB[t][s_][1].to(DEV) for t in G]
        if m in ("ours", "ours_pgc"):
            return getattr(MM, m)(Bs, As, p["alpha"], U.USIG[s_].to(DEV), U.VS[s_].to(DEV))
        if m == "ours_lgo":
            out = set(G) | {DUP[t] for t in G if t in DUP}
            rest = [t for t in range(76) if t not in out]
            return MM.ours_lgo(Bs, As, p["alpha"], [self.AB[t][s_][0].to(DEV) for t in rest], [self.AB[t][s_][1].to(DEV) for t in rest])
        if m == "dare_ties":
            return MM.dare_ties(Bs, As, salt=zlib.crc32(s_.encode()) % 100003, **p)
        return getattr(MM, m)(Bs, As, **p)
    def build(self, m, p, G, out):
        sd = dict(self.base_sd)
        with torch.no_grad():
            for s_, ck in self.ck.items():
                D = self.delta(m, p, G, s_).float()
                W = self.base_sd[ck]
                D = D if D.shape == W.shape else D.T
                sd[ck] = (W.to(DEV).float() + D).to(W.dtype).cpu()
        shutil.rmtree(out, ignore_errors=True); (out / "assets/physical-intelligence/libero").mkdir(parents=True)
        st.save_file(sd, str(out / "model.safetensors"))
        shutil.copy2(BASE_DIR / "config.json", out / "config.json")
        shutil.copy2(REPO / "results/analysis/norm_assets/norm_stats.json", out / "assets/physical-intelligence/libero/norm_stats.json")
        torch.cuda.empty_cache()

def plan_file(jobs, n, offset, tag_):
    for j in jobs: j.update(suite="libero_90", norm_stats=U.FLEET_NORM)
    p = REPO / f"sweep/_e110_{tag_}_{os.getpid()}.json"
    meta = dict(U.META)
    if os.environ.get("E110_GPUS"):                    # e.g. "0,1": keep a GPU free for other users (10-08: shiva GPU 2)
        gpus = [int(x) for x in os.environ["E110_GPUS"].split(",")]
        n = 3 * len(gpus); start = (list(meta.get("port_per_worker") or [8000]))[0]
        meta["n_workers"] = n; meta["gpu_per_worker"] = [gpus[i % len(gpus)] for i in range(n)]
        meta["port_per_worker"] = [start + i for i in range(n)]       # orchestrator indexes PORTS[worker]
    json.dump(dict(meta, n_trials=n, state_offset=offset, jobs=jobs, eval_script="examples/libero/main_dense.py",
                   save_videos="0", heavy_parallel=2), open(p, "w"), indent=1)
    return p

def run_items(items, n, offset, stage):
    """items: [(m, p, i, G, panel_tasks)]; builds round r+1 while round r is evaluated"""
    B = Builder(); rounds = [items[r:r + ROUND] for r in range(0, len(items), ROUND)]
    def build_round(rd):
        js = []
        for m, p, i, G, pt in rd:
            t1 = time.time(); out = PRE / job(m, p, K, i, stage); B.build(m, p, G, out)
            print(f"[e110] built {job(m, p, K, i, stage)} ({len(G)} tasks) in {time.time()-t1:.0f}s", flush=True)
            js += [{"name": sjob(m, p, K, i, stage, t), "method": "average", "params": {}, "adapters": [], "adapter_paths": [],
                    "prebuilt_dir": str(out), "eval_scenes": [SC[t][0]]} for t in pt
                   if rec(sjob(m, p, K, i, stage, t), t) is None]
        return js
    cur = build_round(rounds[0]) if rounds else []
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                                 str(plan_file(cur, n, offset, f"{stage}{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e110] {stage} round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def val_score(m, p):
    s = n = 0
    for i, G, pt in panel_groups(K):
        for t in pt:
            r = rec(sjob(m, p, K, i, "val", t), t)
            if r is None: return None
            s += r["n_success"]; n += r["n_trials"]
    return s / n

def main():
    stage = os.environ.get("E110_STAGE", "val"); U._clear_orphaned_builds()
    methods = os.environ["E110_METHODS"].split(",") if os.environ.get("E110_METHODS") else list(grid(K))
    PG = panel_groups(K)
    if stage == "val":
        items = [(m, p, i, G, pt) for m in methods for p in grid(K)[m] for i, G, pt in PG
                 if any(rec(sjob(m, p, K, i, "val", t), t) is None for t in pt)]
        if os.environ.get("E110_SHARD"):                                      # e.g. "0,1,2/4" (shiva) and "3/4" (odin)
            # shard over the FIXED list of the 10-08 extension (the 22 new settings x groups, grid order), so a restart
            # that sees fewer remaining items keeps the same split (positions in the remaining list would shift)
            keep, S_ = os.environ["E110_SHARD"].split("/"); keep = {int(x) for x in keep.split(",")}
            orig = [(tag(m, p), i) for m in methods for p in grid(K)[m] for i, G, pt in PG if tag(m, p) not in OLD_TAGS]
            mine = {key for j, key in enumerate(orig) if j % int(S_) in keep}
            items = [it for it in items if (tag(it[0], it[1]), it[2]) in mine]
        print(f"[e110] val k={K}: {len(items)} merged models ({len(methods)} methods)", flush=True)
        run_items(items, 10, 25, "val")
    elif stage == "test":
        items = []
        for m in methods:
            sc = {tag(m, p): (val_score(m, p), p) for p in grid(K)[m]}
            if any(v[0] is None for v in sc.values()): print(f"[e110] {m}: validation incomplete, skipped", flush=True); continue
            best = max(grid(K)[m], key=lambda p: sc[tag(m, p)][0])
            print(f"[e110] {m}: best {tag(m, best)} (val {sc[tag(m, best)][0]:.3f}; grid " + ", ".join(f"{k} {v[0]:.3f}" for k, v in sc.items()) + ")", flush=True)
            items += [(m, best, i, G, pt) for i, G, pt in PG if any(rec(sjob(m, best, K, i, "test", t), t) is None for t in pt)]
        print(f"[e110] test k={K}: {len(items)} merged models", flush=True)
        run_items(items, 25, 0, "test")
    elif stage == "report":
        from scipy import stats
        base = {t: rec(f"e109_base_t{t}", t) for t in PANEL}
        L = [f"=== EXP-110 multi-task merging benchmark, k={K}: best validation setting per method, test on states 0..24 (25 eps/task) ===",
             f"{'method':12s} {'best setting':24s} {'val':>6s} {'test':>6s} {'retention':>9s} {'vs avg pp':>9s} {'p':>6s} {'vs ours pp':>10s} {'p':>6s} {'tasks <-30pp':>12s}"]
        res = {}
        for m in methods:
            sc = {tag(m, p): val_score(m, p) for p in grid(K)[m]}
            if any(v is None for v in sc.values()): continue
            best = max(grid(K)[m], key=lambda p: sc[tag(m, p)])
            per = {}
            for i, G, pt in PG:
                for t in pt:
                    r = rec(sjob(m, best, K, i, "test", t), t)
                    if r is not None: per[t] = r["success_rate"]
            res[m] = (best, sc[tag(m, best)], per)
        S = {t: base[t]["success_rate"] for t in PANEL if base[t]}
        def cmp(a, b):
            T_ = sorted(set(res[a][2]) & set(res[b][2])); d = np.array([res[a][2][t] - res[b][2][t] for t in T_]) * 100
            return (d.mean(), stats.wilcoxon(d).pvalue if len(d) > 1 and np.any(d) else float("nan")) if len(d) else (float("nan"), float("nan"))
        for m, (best, v, per) in sorted(res.items(), key=lambda x: -np.mean(list(x[1][2].values())) if x[1][2] else 0):
            if not per: L.append(f"{m:12s} {tag(m, best):24s} {100*v:5.1f}%   (test pending)"); continue
            T_ = [t for t in per if t in S]; mt = np.mean([per[t] for t in T_]); st_ = np.mean([S[t] for t in T_])
            da, pa = cmp(m, "avg") if "avg" in res and m != "avg" else (float("nan"), float("nan"))
            do, po = cmp(m, "ours") if "ours" in res and m != "ours" else (float("nan"), float("nan"))
            bad = sum(1 for t in T_ if per[t] - S[t] < -0.30)
            L.append(f"{m:12s} {tag(m, best):24s} {100*v:5.1f}% {100*mt:5.1f}% {100*mt/st_:8.1f}% {da:+9.1f} {pa:6.3f} {do:+10.1f} {po:6.3f} {bad:>12d}")
        L.append(f"single-task (EXP-109, same panel/states): {100*np.mean(list(S.values())):.1f}% over {len(S)} tasks")
        txt = "\n".join(L); print(txt); (REPO / f"results/analysis/exp110_report_k{K}.txt").write_text(txt + "\n")
    print(f"[e110] {stage} DONE", flush=True)

if __name__ == "__main__":
    main()
