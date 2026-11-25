#!/usr/bin/env python3
"""EXP-108 — the EXP-107 rule on 30 FOREIGN targets (generality + power), user-approved 10-06.

PRE-REGISTERED (10-06, before any rollout):
  targets: our 30 independently trained (zero-init) LoRAs, 10 each of LIBERO-Goal / Object / Spatial
           (results/analysis/new_loras/{suite}_no_noops__{task}); donors: the 74 unique LIBERO-90 fleet LoRAs.
  rule:    identical to EXP-107 (gate 10 on states 25..34, keep if >= 9/10; r1 8 seeded donors x 5 on 25..29;
           r2 top 3 x 5 on 30..34; r3 top 1 x 5 + target x 5 on 35..39; merge the single donor at full dose c with
           the probe operator only if P(p_donor > p_target | 15 vs 15, Beta(1,1)) >= 0.8).
  scoring: states 0..24 only; merged targets get 4 x 25 = 100 episodes of the merge and 4 x 25 = 100 fresh
           episodes of the target alone; kept targets gain 0 by construction.
  primary: mean gain over the 30 targets (kept = 0). Also reported pooled with EXP-107 (104 targets).
Stages (E108_STAGE): gate | r1 | r2 | r3 | score | report. Records e108_{h6}_{g,g2,r1_d{d},r2_d{d},r3_d{d},m1..m4,b1..b4}.
"""
import hashlib, json, os, pathlib, subprocess, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
UNIQUE = [t for t in range(76) if t not in (18, 62)]
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
CHUNK = int(os.environ.get("E108_CHUNK", "6"))
h6 = lambda n: hashlib.md5(n.encode()).hexdigest()[:6]
TG = []
for d in sorted((REPO / "results/analysis/new_loras").glob("libero_*_no_noops__*")):
    if (d / "adapter_model.safetensors").exists():
        n = d.name; TG.append(dict(name=n, key="np:" + n, path=str(d / "adapter_model.safetensors"),
                                   suite=n.split("__")[0].replace("_no_noops", ""), task=n.split("__", 1)[1], tag=h6(n)))
for g in TG: U.EXTPATH[g["key"]] = g["path"]

def rec(job, g):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists(): return None
    for ln in open(f):
        r = json.loads(ln)
        if r.get("status") == "ok" and r["scene"] == g["task"]: return r
    return None
def succ(job, g):
    r = rec(job, g); return None if r is None else (int(r["n_success"]), int(r["n_trials"]), float(np.mean(r.get("progress", [0]))))
J_ = lambda g, s: f"e108_{g['tag']}_{s}"

def donors(g): return sorted(int(d) for d in np.random.default_rng(int(g["tag"], 16)).choice(UNIQUE, 8, replace=False))
def gated_in(g): x = succ(J_(g, "g"), g); return x is not None and x[0] < 9
def rank(g, ds, stages):
    def key(d):
        s = [succ(J_(g, f"{st}_d{d}"), g) for st in stages]; s = [x for x in s if x]
        return (sum(x[0] for x in s) / max(1, sum(x[1] for x in s)), np.mean([x[2] for x in s]) if s else 0, -d)
    return sorted(ds, key=key, reverse=True)
def top3(g): return rank(g, donors(g), ["r1"])[:3]
def top1(g): return rank(g, top3(g), ["r1", "r2"])[0]
def decision(g):
    if not gated_in(g): return None, "gate: kept"
    d = top1(g); s = [succ(J_(g, f"{st}_d{d}"), g) for st in ("r1", "r2", "r3")]
    a, b = succ(J_(g, "g"), g), succ(J_(g, "g2"), g)
    if any(x is None for x in s) or b is None: return None, "incomplete"
    sd, nd = sum(x[0] for x in s), sum(x[1] for x in s); st_, nt = a[0] + b[0], a[1] + b[1]
    rng = np.random.default_rng(108)
    p = float((rng.beta(1 + sd, 1 + nd - sd, 200000) > rng.beta(1 + st_, 1 + nt - st_, 200000)).mean())
    return (d if p >= 0.8 else None), f"donor {d} {sd}/{nd} vs target {st_}/{nt}: P(better) {p:.2f}"

def run(jobs, n, offset, tag):
    """jobs: [(name, g, ('build', d) | path)]; per suite, in chunks (odin disk / shiva quota)"""
    for su in sorted({g["suite"] for _, g, _ in jobs}):
        js = [j for j in jobs if j[1]["suite"] == su]
        for i in range(0, len(js), CHUNK):
            built, J = [], []
            for name, g, src in js[i:i + CHUNK]:
                if isinstance(src, tuple):
                    p = U.build(name, g["key"], [(src[1], REF / U.resid_norm(src[1]))]); built.append(p)
                else: p = src
                J.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [str(p)],
                          "eval_scenes": [g["task"]], "suite": su, "norm_stats": U.FLEET_NORM})
            planp = REPO / f"sweep/_e108_{tag}_{os.getpid()}_{su}_{i}.json"
            json.dump(dict(U.META, n_trials=n, state_offset=offset, jobs=J, eval_script="examples/libero/main_dense.py",
                           save_videos="0", heavy_parallel=2), open(planp, "w"), indent=1)
            subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)
            for p in built: pathlib.Path(p).unlink(missing_ok=True)
            print(f"[e108] {tag} {su}: chunk {i // CHUNK + 1}/{(len(js) + CHUNK - 1) // CHUNK} done", flush=True)

def main():
    stage = os.environ.get("E108_STAGE", "gate"); U._clear_orphaned_builds()
    if stage == "gate":
        J = [(J_(g, "g"), g, g["path"]) for g in TG if rec(J_(g, "g"), g) is None]
        print(f"[e108] gate: {len(TG)} foreign targets, {len(J)} jobs x 10", flush=True); run(J, 10, 25, "gate")
    elif stage == "r1":
        J = [(J_(g, f"r1_d{d}"), g, ("build", d)) for g in TG if gated_in(g) for d in donors(g) if rec(J_(g, f"r1_d{d}"), g) is None]
        print(f"[e108] r1: {sum(gated_in(g) for g in TG)} gated in; {len(J)} jobs x 5", flush=True); run(J, 5, 25, "r1")
    elif stage == "r2":
        J = [(J_(g, f"r2_d{d}"), g, ("build", d)) for g in TG if gated_in(g) for d in top3(g) if rec(J_(g, f"r2_d{d}"), g) is None]
        print(f"[e108] r2: {len(J)} jobs x 5", flush=True); run(J, 5, 30, "r2")
    elif stage == "r3":
        J = [(J_(g, f"r3_d{top1(g)}"), g, ("build", top1(g))) for g in TG if gated_in(g) if rec(J_(g, f"r3_d{top1(g)}"), g) is None]
        J += [(J_(g, "g2"), g, g["path"]) for g in TG if gated_in(g) if rec(J_(g, "g2"), g) is None]
        print(f"[e108] r3: {len(J)} jobs x 5", flush=True); run(J, 5, 35, "r3")
    elif stage == "score":
        J = []
        for g in TG:
            d, _ = decision(g)
            if d is None: continue
            J += [(J_(g, f"m{k}"), g, ("build", d)) for k in range(1, 5) if rec(J_(g, f"m{k}"), g) is None]
            J += [(J_(g, f"b{k}"), g, g["path"]) for k in range(1, 5) if rec(J_(g, f"b{k}"), g) is None]
        print(f"[e108] score: {len(J)} jobs x 25 on states 0..24", flush=True); run(J, 25, 0, "score")
    else:
        L = ["=== EXP-108 live: the EXP-107 rule on 30 foreign targets; scored on states 0..24 (100 vs 100 episodes) ===",
             f"{'target':58s} {'gate':>5} | decision | merge | target | gain"]
        gains, gm, harmed = [], [], 0
        for g in TG:
            x = succ(J_(g, "g"), g)
            if x is None: L.append(f"{g['name'][:58]:58s} gate missing"); continue
            d, why = decision(g)
            if d is None: gains.append(0.0); L.append(f"{g['name'][:58]:58s} {x[0]:>2}/10 | keep ({why})"); continue
            m = [succ(J_(g, f"m{k}"), g) for k in range(1, 5)]; b = [succ(J_(g, f"b{k}"), g) for k in range(1, 5)]
            if any(v is None for v in m + b): L.append(f"{g['name'][:58]:58s} merge d{d}: scoring incomplete"); continue
            pm = sum(v[0] for v in m) / sum(v[1] for v in m); pb = sum(v[0] for v in b) / sum(v[1] for v in b)
            gn = 100 * (pm - pb); gains.append(gn); gm.append(gn); harmed += gn < -8
            L.append(f"{g['name'][:58]:58s} {x[0]:>2}/10 | merge d{d} ({why}) | {100*pm:5.1f} | {100*pb:5.1f} | {gn:+5.1f}")
        gains = np.array(gains)
        L.append(f"\nPRIMARY: mean gain over {len(gains)} foreign targets (kept = 0): {gains.mean():+.2f}pp "
                 f"(SE {gains.std(ddof=1)/np.sqrt(len(gains)):.2f}; t-test p={stats.ttest_1samp(gains, 0).pvalue:.3f})")
        if gm: L.append(f"merged: {len(gm)}; mean gain when merged {np.mean(gm):+.2f}pp; harmed (< -8pp) {harmed}/{len(gm)}")
        txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp108_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
