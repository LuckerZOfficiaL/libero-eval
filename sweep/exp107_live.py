#!/usr/bin/env python3
"""EXP-107 — live, clean-protocol, powered test of the gated, certified successive-halving rule (fleet-wide).

PRE-REGISTERED (10-06, before any rollout):
  rule per target t (budget <= 80 rollouts, all on initial states 25..49):
    gate   target alone x 10 (states 25..34); >= 9/10 successes -> keep the target (no search)
    r1     8 seeded-random donors x 5 (states 25..29), merged with the probe operator W = S16 + r_t + c r_d (full c)
    r2     top 3 by r1 x 5 more (states 30..34)
    r3     top 1 by r1+r2 x 5 more (states 35..39) + target x 5 more (states 35..39)
    decide merge the r3 donor only if P(p_donor > p_target | 15 vs 15 rollouts, Beta(1,1)) >= 0.8; else keep target
  scoring on states 0..24 (never used for any decision): each merged target -> merge 2 x 25 episodes, target alone
    25 fresh + e92_base_f{t} (25, independent run) = 50 vs 50. Kept targets: gain = 0 by construction.
  primary endpoint: mean gain over ALL 74 targets (kept = 0); secondary: mean gain over merged targets, and the
    fraction of merged targets that are harmed (< -8pp).
  offline expectation (EXP-107 sim, weak targets): +4.0pp overall, +13pp when merged, harmful ~3%.
Stages (E107_STAGE): gate | r1 | r2 | r3 | score | report. Records e107_t{t}_{g,g2,r1_d{d},r2_d{d},r3_d{d},m,m2,b}.
"""
import json, os, pathlib, subprocess, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
UNIQUE = [t for t in range(76) if t not in (18, 62)]
DUP = {7: 18, 18: 7, 29: 62, 62: 29}
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
TGTS = [int(x) for x in os.environ["E107_TARGETS"].split(",")] if os.environ.get("E107_TARGETS") else UNIQUE
CHUNK = int(os.environ.get("E107_CHUNK", "18"))               # merge builds per chunk (0.9 GB each; shiva quota 46 GB)

def rec(job, t):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists(): return None
    for ln in open(f):
        r = json.loads(ln)
        if r.get("status") == "ok" and r["scene"] == SC[t][0]: return r
    return None
def succ(job, t):
    r = rec(job, t); return None if r is None else (int(r["n_success"]), int(r["n_trials"]), float(np.mean(r.get("progress", [0]))))

def donors(t):
    pool = [d for d in UNIQUE if d != t and d != DUP.get(t, -1)]
    return sorted(int(d) for d in np.random.default_rng(10700 + t).choice(pool, 8, replace=False))
def gated_in(t):
    g = succ(f"e107_t{t}_g", t); return g is not None and g[0] < 9
def rank(t, ds, stages):
    def key(d):
        s = [succ(f"e107_t{t}_{st}_d{d}", t) for st in stages]; s = [x for x in s if x]
        return (sum(x[0] for x in s) / max(1, sum(x[1] for x in s)), np.mean([x[2] for x in s]) if s else 0, -d)
    return sorted(ds, key=key, reverse=True)
def top3(t): return rank(t, donors(t), ["r1"])[:3]
def top1(t): return rank(t, top3(t), ["r1", "r2"])[0]
def decision(t):
    if not gated_in(t): return None, "gate: kept"
    d = top1(t); s = [succ(f"e107_t{t}_{st}_d{d}", t) for st in ("r1", "r2", "r3")]
    g, g2 = succ(f"e107_t{t}_g", t), succ(f"e107_t{t}_g2", t)
    if any(x is None for x in s) or g2 is None: return None, "incomplete"
    sd, nd = sum(x[0] for x in s), sum(x[1] for x in s); st_, nt = g[0] + g2[0], g[1] + g2[1]
    rng = np.random.default_rng(107)
    p = float((rng.beta(1 + sd, 1 + nd - sd, 200000) > rng.beta(1 + st_, 1 + nt - st_, 200000)).mean())
    return (d if p >= 0.8 else None), f"donor {d} {sd}/{nd} vs target {st_}/{nt}: P(better) {p:.2f}"

def run(jobs, n, offset, tag, build=None):
    """jobs: [(name, t, adapter_path or None-to-build-from (t, d))]; builds in chunks to respect the quota"""
    for i in range(0, len(jobs), CHUNK):
        part, built, J = jobs[i:i + CHUNK], [], []
        for name, t, src in part:
            if isinstance(src, tuple):
                p = U.build(f"{name}", t, [(src[1], REF / U.resid_norm(src[1]))]); built.append(p)
            else: p = src
            J.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [str(p)],
                      "eval_scenes": [SC[t][0]], "suite": "libero_90", "norm_stats": U.FLEET_NORM})
        planp = REPO / f"sweep/_e107_{tag}_{os.getpid()}_{i}.json"
        json.dump(dict(U.META, n_trials=n, state_offset=offset, jobs=J, eval_script="examples/libero/main_dense.py",
                       save_videos="0", heavy_parallel=2), open(planp, "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[e107] {tag}: chunk {i // CHUNK + 1}/{(len(jobs) + CHUNK - 1) // CHUNK} done", flush=True)

def main():
    stage = os.environ.get("E107_STAGE", "gate"); U._clear_orphaned_builds()
    if stage == "gate":
        J = [(f"e107_t{t}_g", t, U.afile[t]) for t in TGTS if rec(f"e107_t{t}_g", t) is None]
        print(f"[e107] gate: {len(J)} target-alone jobs x 10", flush=True); run(J, 10, 25, "gate")
    elif stage == "r1":
        J = [(f"e107_t{t}_r1_d{d}", t, (t, d)) for t in TGTS if gated_in(t) for d in donors(t) if rec(f"e107_t{t}_r1_d{d}", t) is None]
        print(f"[e107] r1: {sum(gated_in(t) for t in TGTS)} targets gated in; {len(J)} donor jobs x 5", flush=True); run(J, 5, 25, "r1")
    elif stage == "r2":
        J = [(f"e107_t{t}_r2_d{d}", t, (t, d)) for t in TGTS if gated_in(t) for d in top3(t) if rec(f"e107_t{t}_r2_d{d}", t) is None]
        print(f"[e107] r2: {len(J)} donor jobs x 5", flush=True); run(J, 5, 30, "r2")
    elif stage == "r3":
        J = [(f"e107_t{t}_r3_d{top1(t)}", t, (t, top1(t))) for t in TGTS if gated_in(t) if rec(f"e107_t{t}_r3_d{top1(t)}", t) is None]
        J += [(f"e107_t{t}_g2", t, U.afile[t]) for t in TGTS if gated_in(t) if rec(f"e107_t{t}_g2", t) is None]
        print(f"[e107] r3: {len(J)} jobs x 5", flush=True); run(J, 5, 35, "r3")
    elif stage == "score":
        J = []
        for t in TGTS:
            d, why = decision(t)
            if d is None: continue
            for nm in ("m", "m2"):
                if rec(f"e107_t{t}_{nm}", t) is None: J.append((f"e107_t{t}_{nm}", t, (t, d)))
            if rec(f"e107_t{t}_b", t) is None: J.append((f"e107_t{t}_b", t, U.afile[t]))
        print(f"[e107] score: {len(J)} jobs x 25 on states 0..24", flush=True); run(J, 25, 0, "score")
    else:
        L = ["=== EXP-107 live: gated, certified successive halving; scored on initial states 0..24 ===",
             f"{'t':>4} {'gate':>5} | decision | merge (50 ep) | target (50 ep) | gain"]
        gains, gm, harmed, nroll = [], [], 0, 0
        for t in TGTS:
            g = succ(f"e107_t{t}_g", t)
            if g is None: continue
            d, why = decision(t)
            nroll += 10 + (0 if not gated_in(t) else 40 + 15 + 5 + 5)
            if d is None: gains.append(0.0); L.append(f"{t:>4} {g[0]:>2}/10 | keep ({why})"); continue
            m = [succ(f"e107_t{t}_{nm}", t) for nm in ("m", "m2")]; b = succ(f"e107_t{t}_b", t)
            e92 = rec(f"e92_base_f{t}", t)
            if any(x is None for x in m) or b is None or e92 is None: L.append(f"{t:>4} merge {d}: scoring incomplete"); continue
            pm = sum(x[0] for x in m) / sum(x[1] for x in m); pb = (b[0] + e92["n_success"]) / (b[1] + e92["n_trials"])
            gn = 100 * (pm - pb); gains.append(gn); gm.append(gn); harmed += gn < -8
            L.append(f"{t:>4} {g[0]:>2}/10 | merge d{d} ({why}) | {100*pm:5.1f} | {100*pb:5.1f} | {gn:+5.1f}")
        gains, gm = np.array(gains), np.array(gm)
        L.append(f"\nPRIMARY: mean gain over {len(gains)} targets (kept = 0): {gains.mean():+.2f}pp "
                 f"(SE {gains.std(ddof=1)/np.sqrt(len(gains)):.2f}; one-sample t p={stats.ttest_1samp(gains, 0).pvalue:.3f})")
        if len(gm):
            L.append(f"merged targets: {len(gm)}; mean gain when merged {gm.mean():+.2f}pp (SE {gm.std(ddof=1)/np.sqrt(len(gm)) if len(gm)>1 else float('nan'):.2f}; "
                     f"Wilcoxon p={stats.wilcoxon(gm).pvalue if len(gm) > 1 and np.any(gm) else float('nan'):.3f}); harmed (< -8pp) {harmed}/{len(gm)}")
        L.append(f"rollouts spent on decisions: {nroll} ({nroll/len(gains):.1f} per target)")
        txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp107_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
