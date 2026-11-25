#!/usr/bin/env python3
"""EXP-120 — PRIVILEGED TASKS: protect a chosen set of tasks in continual merging (user idea 10-16).

Idea (user): the tasks we want to preserve are the ones that define the shared component; the set can be changed at any
time (the merge is sum-based, so a new set only means recomputing the shared component from the privileged adapters).
So far the privileged set was the LIBERO-90 library (its rank-16 shared part protected LIBERO-90 in EXP-116). Within
LIBERO-90 every adapter shares the same 16 directions, so privileging a subset of LIBERO-90 cannot be selective; the
selective case is a privileged task with its own structure. A one-task group's shared component is the task's own
task-specific part (its residual w.r.t. the library's shared part; rank-16 SVD of one rank-16 matrix = itself).

PRE-REGISTERED (10-16, before any rollout):
  model      W = base + S16(LIBERO-90) + w * sum_{p in P} r_p + (1/k) * sum_{i not in P} r_i
  P          the first 2 new tasks of each sequence (arrival order; fixed rule, no selection by results)
  arms       w = 1 (privileged at full strength) and w = 0.25
  conditions scratch (EXP-116 sequences) and anchored (EXP-118 sequences); t = 8 (24 tasks); same evaluated tasks as EXP-116
             (8 LIBERO-90 + all 8 new), 10 episodes, states 0..9; reference = ours_lib at t = 8 (EXP-116 / EXP-118 records)
  readout    retention of privileged new tasks, non-privileged new tasks, LIBERO-90 tasks, each vs ours_lib (paired)
  predictions privileged tasks are protected far more than under ours_lib (selectivity); the cost lands on LIBERO-90 and on the
             other new tasks, larger for scratch adapters and for w = 1 (EXP-117: up-weighted foreign adapters interfere)
Env: E120_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp116_continual as C
import exp118_anchored as A
import exp113_controls as X
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U
WS, T, NP = (1.0, 0.25), 8, 2
COND = {"z": ("c116", C.sequence), "a": ("c118", A.sequence)}

class PBuilder(C.CBuilder):
    def _delta(self, m, p, members, s_):
        if m != "priv": return super()._delta(m, p, members, s_)
        F = [self.fac(i) for i in members]
        if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F): return None
        k = len(members); w = [p["w"] if i in p["P"] else 1.0 / k for i in members]
        Bs = [wi * f[s_][0].to(E.DEV) for wi, f in zip(w, F)]; As = [f[s_][1].to(E.DEV) for f in F]
        return MM.ours(Bs, As, 1.0, U.USIG[s_].to(E.DEV), U.VS[s_].to(E.DEV))

def items():
    out = []
    for cd, (_, seqf) in COND.items():
        for s in range(C.NSEQ):
            old, new = seqf(s); mem = old + new[:T]; ev = sorted(old[:C.NEVAL] + new[:T])
            for w in WS: out.append((f"c120{cd}s{s}t{T}-priv_w{w:g}", s, {"w": w, "P": new[:NP]}, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[3]), "test", it[1], it[4]) for sc in scs)]
    print(f"[e120] {len(todo)} merged models", flush=True)
    if not todo: return
    B = PBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, p, mem, ev in rd:
            out = E.PRE / f"e120_{tg}"; t1 = time.time(); B.build("priv", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e120] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e120_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e120] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def groups(tg, s, seqf):
    """retention per task, split into old / privileged / other new; keyed by (sequence, task index)"""
    old, new = seqf(s); ev = sorted(old[:C.NEVAL] + new[:T]); sg = X.single()
    sc = E.member_success(tg, len(old) + T, "test", s, ev)
    if len(sc) < len(ev): return None
    r = {i: sc[i] / sg[i] for i in ev if sg.get(i, 0) > 0}
    return {"old": {(s, i): v for i, v in r.items() if i < X.NF}, "priv": {(s, i): v for i, v in r.items() if i in new[:NP]},
            "other new": {(s, i): v for i, v in r.items() if i in new[NP:]}}

def report():
    from scipy import stats
    L = ["=== EXP-120 privileged tasks (first 2 new tasks at weight w; others 1/k; LIBERO-90 shared part), t = 8, retention ==="]
    for cd, (pref, seqf) in COND.items():
        ref = [groups(f"{pref}s{s}t{T}-ours_lib", s, seqf) for s in range(C.NSEQ)]
        for w in WS:
            arm = [groups(f"c120{cd}s{s}t{T}-priv_w{w:g}", s, seqf) for s in range(C.NSEQ)]
            if any(a is None for a in arm + ref): L.append(f"{cd} w={w:g}: incomplete"); continue
            L.append(f"{'scratch' if cd == 'z' else 'anchored'}, w = {w:g}:")
            for g in ("priv", "old", "other new"):
                x = {k_: v for a in arm for k_, v in a[g].items()}; y = {k_: v for a in ref for k_, v in a[g].items()}
                ks = [q for q in x if q in y]; d = np.array([100 * (x[q] - y[q]) for q in ks])
                p = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                L.append(f"  {g:9s}: {100*np.mean([x[q] for q in ks]):5.1f}% vs ours_lib {100*np.mean([y[q] for q in ks]):5.1f}%  "
                         f"({d.mean():+6.1f}pp, better {int((d>0).sum())} / worse {int((d<0).sum())}, p={p:.4f}, n={len(ks)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp120_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E120_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e120] {st} DONE", flush=True)
