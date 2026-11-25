#!/usr/bin/env python3
"""EXP-111 — powered confirmatory test: shared part + half the mean residual vs plain averaging, one model of all 74 tasks.

PRE-REGISTERED (10-09, user-approved, before any rollout):
  arms     avg  : W = base + (1/74) sum_i D_i
           ours : W = base + S16 + alpha * sum_i r_i with the FIXED rule alpha = 1/(2k) = 1/148
                  (S16 = pool shared rank-16 part; r_i = two-sided residuals; rule taken from EXP-110 k=74 validation)
  tasks    ALL 74 unique LIBERO-90 tasks (the EXP-110 test used only a 24-task panel)
  episodes 100 per task per arm = 4 repetitions x 25 on initial states 0..24 (never used for any decision)
  primary  mean paired difference ours - avg over the 74 tasks (each task pooled over its 100 episodes),
           two-sided paired t-test, alpha 0.05; Wilcoxon signed-rank reported alongside
  secondary retention vs single-task (e92_base_f{t}, 25 eps, independent earlier run); tasks losing > 30pp per arm
Records e111_{arm}_t{t}_r{r}. Env: E111_STAGE run|report, E111_GPUS (0,1,2), E111_DEV (cuda:1).
"""
import json, os, pathlib, shutil, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
UNIQUE = [t for t in range(76) if t not in (18, 62)]
ARMS = {"avg": ("avg", {}), "ours": ("ours", {"alpha": 1 / 148})}
PRE = pathlib.Path("/dev/shm/mergesweep/prebuilt")
REPS = 4

def rec(job, t):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists(): return None
    for ln in open(f):
        r = json.loads(ln)
        if r.get("status") == "ok" and r["scene"] == SC[t][0]: return r
    return None

def main():
    stage = os.environ.get("E111_STAGE", "run")
    if stage == "run":
        os.environ.setdefault("E110_DEV", os.environ.get("E111_DEV", "cuda:1"))
        import exp110_bench as E
        U._clear_orphaned_builds()
        B = E.Builder(); dirs = {}
        for arm, (m, p) in ARMS.items():
            out = PRE / f"e111_{arm}"; B.build(m, p, UNIQUE, out); dirs[arm] = str(out)
            print(f"[e111] built {arm}", flush=True)
        jobs = [{"name": f"e111_{arm}_t{t}_r{r}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [],
                 "prebuilt_dir": dirs[arm], "eval_scenes": [SC[t][0]], "suite": "libero_90", "norm_stats": U.FLEET_NORM}
                for r in range(1, REPS + 1) for t in UNIQUE for arm in ARMS if rec(f"e111_{arm}_t{t}_r{r}", t) is None]
        meta = dict(U.META)
        gpus = [int(x) for x in os.environ.get("E111_GPUS", "0,1,2").split(",")]
        n = 3 * len(gpus); start = (list(meta.get("port_per_worker") or [8000]))[0]
        meta.update(n_workers=n, gpu_per_worker=[gpus[i % len(gpus)] for i in range(n)], port_per_worker=[start + i for i in range(n)])
        planp = REPO / f"sweep/_e111_{os.getpid()}.json"
        json.dump(dict(meta, n_trials=25, state_offset=0, jobs=jobs, eval_script="examples/libero/main_dense.py",
                       save_videos="0", heavy_parallel=2), open(planp, "w"), indent=1)
        print(f"[e111] {len(jobs)} jobs x 25 episodes on {n} workers", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)
        for d in dirs.values(): shutil.rmtree(d, ignore_errors=True)
        print("[e111] run DONE", flush=True); return
    from scipy import stats
    S = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for t in UNIQUE:
            rs = [rec(f"e111_{arm}_t{t}_r{r}", t) for r in range(1, REPS + 1)]; rs = [x for x in rs if x]
            if rs: S[arm][t] = (sum(x["n_success"] for x in rs), sum(x["n_trials"] for x in rs))
    T = [t for t in UNIQUE if t in S["avg"] and t in S["ours"]]
    pa = np.array([S["avg"][t][0] / S["avg"][t][1] for t in T]); po = np.array([S["ours"][t][0] / S["ours"][t][1] for t in T])
    d = 100 * (po - pa); ep = [S["ours"][t][1] for t in T]
    base = {t: rec(f"e92_base_f{t}", t) for t in T}; sb = np.array([base[t]["success_rate"] if base[t] else np.nan for t in T])
    L = [f"=== EXP-111 confirmatory test (pre-registered): one model of all 74 tasks; {len(T)} tasks, "
         f"{min(ep) if ep else 0}-{max(ep) if ep else 0} episodes per task per arm, states 0..24 ===",
         f"avg  {100*pa.mean():.2f}%   ours (S16 + residuals at 1/(2k))  {100*po.mean():.2f}%",
         f"PRIMARY paired difference ours - avg: {d.mean():+.2f}pp (SE {d.std(ddof=1)/np.sqrt(len(d)):.2f}); "
         f"t-test p={stats.ttest_rel(po, pa).pvalue:.4f}; Wilcoxon p={stats.wilcoxon(d).pvalue if np.any(d) else float('nan'):.4f}; "
         f"tasks up {int((d>0).sum())} / down {int((d<0).sum())} / equal {int((d==0).sum())}",
         f"retention vs single-task (e92, 25 eps, earlier run; {np.isfinite(sb).sum()} tasks): avg {100*np.nanmean(pa)/np.nanmean(sb):.1f}%  ours {100*np.nanmean(po)/np.nanmean(sb):.1f}%",
         f"tasks losing > 30pp vs single-task: avg {int(np.nansum(pa - sb < -0.30))}, ours {int(np.nansum(po - sb < -0.30))}",
         "largest per-task differences: " + ", ".join(f"t{t} {x:+.0f}" for t, x in sorted(zip(T, d), key=lambda z: -abs(z[1]))[:8])]
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp111_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
