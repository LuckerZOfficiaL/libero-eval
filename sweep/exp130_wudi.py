#!/usr/bin/env python3
"""EXP-130 — WUDI-Merging as an additional static, data-free baseline (MergeVLA's strongest; user OK 10-19).

WUDI (Cheng et al., ICML 2025, arXiv 2503.08099): per linear layer minimise sum_i ||(tau_m - tau_i) tau_i^T||^2 / ||tau_i||^2,
tau_m initialised at the SUM of task vectors, Adam lr 1e-5, 300 iterations, no rescaling; LoRA restored to tau_i = B_i A_i.
Offline check (10-19, write-protected adapters, 24 tasks, 3 module sizes): with the official lr the optimiser barely moves
(tau_m stays ~0.93 x sum = ~22x the norm of averaging; loss orders of magnitude above the loss at averaging); lr 1e-3
reaches the averaging loss level or below; the objective's own optimum has 20-124x averaging's norm (all tasks read the
same shared input directions -> the objective barely constrains scale).

PRE-REGISTERED (10-19, before any rollout):
  setting    EXP-125 (sequences 0-3, every new task write-protected, t = 4, 8, same evaluated tasks, 10 eps, states 0..9)
  arms       wudi_default  : official (lr 1e-5, 300 it, sum init, no rescaling)
             wudi_conv     : lr 1e-3, 300 it, no rescaling (optimiser converged to the objective's loss level)
             wudi_nm       : wudi_conv rescaled to averaging's norm x 1.33 (the convention used for TIES / DARE / MaxAbs)
  readout    old / new retention next to EXP-125's table; ours w=0.25 vs each arm, paired at t = 8
  predictions the unrescaled arms collapse (merged update ~12-22x averaging's norm); wudi_nm behaves like the other linear
             merges (new tasks ~20%); none reaches ours on new tasks
Env: E130_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp124_wpcontinual as W
import exp120_privileged as P
import exp116_continual as C
import exp112_bench as E
import unified_pool_driver as U
ARMS = [("wudi_default", {"iters": 300, "lr": 1e-5}, None), ("wudi_conv", {"iters": 300, "lr": 1e-3}, None),
        ("wudi_nm", {"iters": 300, "lr": 1e-3}, 1.33)]

class WBuilder(P.PBuilder):
    def _delta(self, m, p, members, s_):
        return super()._delta("wudi" if m.startswith("wudi") else m, p, members, s_)

def items():
    out = []
    for s in W.SEQS:
        old, new = W.sequence(s)
        for t in W.TS:
            mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t])
            for nm, p, c in ARMS: out.append((f"c130s{s}t{t}-{nm}", s, t, nm, p, c, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[7]) for sc in scs)]
    print(f"[e130] {len(todo)} merged models", flush=True)
    if not todo: return
    B = WBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, m, p, c, mem, ev in rd:
            out = E.PRE / f"e130_{tg}"; t1 = time.time(); B.build(m, p, c, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e130] built {tg} ({len(mem)} tasks, scale {B.scale(m, p, c, mem):.3f}) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e130_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e130] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    sg = W.single()
    rows = [(nm, f"c130s{{s}}t{{t}}-{nm}") for nm, _, _ in ARMS] + [("ours w=0.25", "c124s{s}t{t}-wp_w0.25"), ("averaging", "c125s{s}t{t}-avg"),
                                                                   ("OPCM", "c125s{s}t{t}-opcm")]
    L = ["=== EXP-130 WUDI on the write-protected adapters (sequences 0-3), retention old / new ==="]
    for t in W.TS:
        G = {}
        for nm, f in rows:
            g = [W.groups(f.format(s=s, t=t), s, t, W.sequence, sg) for s in W.SEQS]
            if any(x is None for x in g): L.append(f"t={t} {nm:13s} incomplete"); continue
            G[nm] = g
            L.append(f"t={t} {nm:13s} old {100*np.mean([v for x in g for v in x['old'].values()]):5.1f}%  new {100*np.mean([v for x in g for v in x['new'].values()]):5.1f}%")
        if t == 8 and "ours w=0.25" in G:
            for nm, _, _ in ARMS:
                if nm not in G: continue
                line = f"  ours w=0.25 vs {nm:12s}"
                for k_ in ("old", "new"):
                    x_ = {q: v for z in G["ours w=0.25"] for q, v in z[k_].items()}; y_ = {q: v for z in G[nm] for q, v in z[k_].items()}; ks = [q for q in x_ if q in y_]
                    d = np.array([100 * (x_[q] - y_[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                    line += f" | {k_} {d.mean():+6.1f}pp ({int((d>0).sum())}/{int((d<0).sum())}, p={p_:.4f})"
                L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp130_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E130_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e130] {st} DONE", flush=True)
