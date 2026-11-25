#!/usr/bin/env python3
"""EXP-106 stage C: final scoring of the two ways to spend 80 rollouts, on initial states 0..24 (never used before).

  probe  deployed rule on the stage-A trials (states 25..34): top-3 donors by success (progress breaks ties), merged
         at c/3 each (W = S16 + r_t + sum c_i/3 r_di); abstain (keep the target) unless the best donor's 10-episode
         success beats the target's own 10 (its first 10 collection episodes, same states)
  sil    the target fine-tuned on its own successful rollouts (exp106_sil.py)
Records e106_t{t}_probe (only when committed), e106_t{t}_sil; 25 episodes each, main_dense.
Then the report: gain vs the target alone (standing baseline R.base, and e92 per-episode on the same 25 states).
Env: E106_STAGE = eval | report
"""
import json, os, pathlib, subprocess, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
import rank_sweep as R
from reinforce_train import scenes as SC
from subset_probe import TARGETS as WEAK
from exp106_collect import probe_donors, REF

def rec(job, t):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists(): return None
    for ln in open(f):
        r = json.loads(ln)
        if r.get("status") == "ok" and r["scene"] == SC[t][0]: return r
    return None

def decision(t):
    s = rec(f"e106_t{t}_self", t)
    tgt = np.mean([x >= 0 for x in s["success_step"][:10]])
    sc = {}
    for d in probe_donors(t):
        r = rec(f"e106_t{t}_pd{d}", t); sc[d] = (r["success_rate"], float(np.mean(r.get("progress", [0]))))
    top = sorted(sc, key=lambda d: sc[d], reverse=True)[:3]
    return top, sc[top[0]][0] > tgt, tgt, sc

REP = os.environ.get("E106_REP", "")                                            # "" = first scoring run, "_r2" = second
def main():
    stage = os.environ.get("E106_STAGE", "eval")
    if stage == "eval":
        U._clear_orphaned_builds(); jobs, built = [], []
        for t in WEAK:
            top, commit, _, _ = decision(t)
            if commit and rec(f"e106_t{t}_probe{REP}", t) is None:
                p = U.build(f"e106_{t}_probe{REP}", t, [(d, REF / U.resid_norm(d) / 3) for d in top]); built.append(p)
                jobs.append({"name": f"e106_t{t}_probe{REP}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [p], "eval_scenes": [SC[t][0]]})
            sp = REPO / f"results/analysis/exp106_sil/t{t}/adapter_model.safetensors"
            if sp.exists() and rec(f"e106_t{t}_sil{REP}", t) is None:
                jobs.append({"name": f"e106_t{t}_sil{REP}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [str(sp)], "eval_scenes": [SC[t][0]]})
        for j in jobs: j["suite"] = "libero_90"; j["norm_stats"] = U.FLEET_NORM
        print(f"[e106c] {len(jobs)} scoring jobs", flush=True)
        if jobs:
            planp = REPO / f"sweep/_e106_final_{os.getpid()}.json"
            json.dump(dict(U.META, n_trials=25, jobs=jobs, eval_script="examples/libero/main_dense.py", save_videos="0", heavy_parallel=2), open(planp, "w"), indent=1)
            subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print("[e106c] DONE", flush=True); return
    L = ["=== EXP-106: two ways to spend 80 rollouts on a weak target; scored on initial states 0..24 (25 episodes) ===",
         f"{'t':>4} {'base':>5} {'self80':>6} | {'probe: action':>14} {'probe':>6} | {'sil':>6} | sil data"]
    G = {"probe": [], "sil": []}
    for t in WEAK:
        b = R.base(t)[0]; s = rec(f"e106_t{t}_self", t)
        top, commit, tgt, sc = decision(t)
        def pooled(job):
            rs = [r for r in (rec(job, t), rec(job + "_r2", t)) if r]
            return float(np.mean([r["success_rate"] for r in rs])) if rs else None
        pv = pooled(f"e106_t{t}_probe") if commit else b
        sv = pooled(f"e106_t{t}_sil")
        meta = REPO / f"results/analysis/exp106_sil/t{t}/sil_meta.json"; m = json.load(open(meta)) if meta.exists() else {}
        if pv is not None: G["probe"].append(100 * (pv - b))
        if sv is not None: G["sil"].append(100 * (sv - b))
        f = lambda v: "  -  " if v is None else f"{100*v:5.1f}"
        L.append(f"{t:>4} {f(b)} {f(s['success_rate'] if s else None):>6} | {('merge ' + ','.join(map(str, top))) if commit else 'abstain':>14} {f(pv):>6} | {f(sv):>6} | "
                 f"{m.get('success_episodes', '-')} succ eps, {m.get('rollout_samples', '-')} samples")
    for k, v in G.items():
        v = np.array(v)
        if len(v): L.append(f"{k:6s}: mean gain {v.mean():+.2f}pp over {len(v)} targets (Wilcoxon p={stats.wilcoxon(v).pvalue if np.any(v) else float('nan'):.3f}); "
                             f"helped {int((v>8).sum())}, hurt {int((v<-8).sum())}")
    if len(G["probe"]) == len(G["sil"]) == len(WEAK):
        dlt = np.array(G["probe"]) - np.array(G["sil"]); L.append(f"probe - sil: {dlt.mean():+.2f}pp (Wilcoxon p={stats.wilcoxon(dlt).pvalue:.3f})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp106_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
