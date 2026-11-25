#!/usr/bin/env python3
"""EXP-125 — every baseline on the SAME write-protected adapters (fairness control for EXP-124, 10-18).

EXP-124: with all new tasks trained write-protected, ours with new tasks at w=0.25 keeps LIBERO-90 at 76.9% and learns the
new tasks to 50.7% after 8 additions — above OPCM (70.9 / 39.1) and ours+OPCM (65.1 / 45.1), but those baselines were run
on the anchored adapters. Adapter type and method are confounded until the baselines see the same adapters.

PRE-REGISTERED (10-18, before any rollout):
  adapters   EXP-124's write-protected adapters for every new task; sequences 0-3; t = 4, 8; same evaluated tasks; 10 eps,
             states 0..9; retention vs the write-protected single-task baselines
  methods    EXP-116's fixed rules, unchanged: avg; ta 1.33/k; ties, dare_ties norm-matched x1.33; cart 0.25/0.08; maxabs
             x1.33; opcm alpha 0.5 (arrival order, norm rescaling)
  readout    old / new retention per method; ours (EXP-124, w = 1/k, 0.1, 0.25) vs each, paired at t = 8
  predictions write-protected adapters help the linear baselines a little (as anchored ones did); no baseline reaches ours at
             w=0.25 on both axes; OPCM on write-protected adapters is the closest competitor
Env: E125_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp124_wpcontinual as W                                         # registers the write-protected adapters
import exp116_continual as C
import exp112_bench as E
import unified_pool_driver as U
METHODS = [x for x in C.METHODS if not x[0].startswith("ours")]

def items():
    out = []
    for s in W.SEQS:
        old, new = W.sequence(s)
        for t in W.TS:
            mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t]); k = len(mem)
            for m, p, c in METHODS:
                if m == "ta": p = {"lam": 1.33 / k}
                out.append((f"c125s{s}t{t}-{m}", s, t, m, p, c, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[7]) for sc in scs)]
    print(f"[e125] {len(todo)} merged models", flush=True)
    if not todo: return
    B = C.CBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, m, p, c, mem, ev in rd:
            out = E.PRE / f"e125_{tg}"; t1 = time.time(); B.build(m, p, c, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e125] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e125_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e125] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    sg = W.single()
    L = ["=== EXP-125 all methods on the write-protected adapters (sequences 0-3), retention old / new ==="]
    for t in W.TS:
        rows = [(m, f"c125s{{s}}t{t}-{m}") for m, _, _ in METHODS] + [(f"ours w={W.wtag(w)}", f"c124s{{s}}t{t}-wp_{W.wtag(w)}") for w in W.WS]
        G = {}
        for nm, f in rows:
            g = [W.groups(f.format(s=s), s, t, W.sequence, sg) for s in W.SEQS]
            if any(x is None for x in g): L.append(f"t={t} {nm:14s} incomplete"); continue
            G[nm] = g
            L.append(f"t={t} {nm:14s} old {100*np.mean([v for x in g for v in x['old'].values()]):5.1f}%  new {100*np.mean([v for x in g for v in x['new'].values()]):5.1f}%")
        if t == 8 and "ours w=w0.25" in G:
            ref = G["ours w=w0.25"]
            for nm, g in G.items():
                if nm.startswith("ours"): continue
                line = f"  ours w=0.25 vs {nm:10s}"
                for k_ in ("old", "new"):
                    x_ = {q: v for a in ref for q, v in a[k_].items()}; y_ = {q: v for a in g for q, v in a[k_].items()}; ks = [q for q in x_ if q in y_]
                    d = np.array([100 * (x_[q] - y_[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                    line += f" | {k_} {d.mean():+6.1f}pp ({int((d>0).sum())}/{int((d<0).sum())}, p={p_:.4f})"
                L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp125_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E125_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e125] {st} DONE", flush=True)
