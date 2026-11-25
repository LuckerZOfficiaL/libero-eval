#!/usr/bin/env python3
"""EXP-121 — WRITE-PROTECTED privileged tasks (follow-up to EXP-120, 10-17).

EXP-120: privileging 2 anchored new tasks (their task-specific parts at weight w) protects them (w=0.25: 65% vs 30%) but
costs LIBERO-90 14pp (w=1: 57pp). Offline energy split (10-17, 6 adapters per group, LIBERO-90 shared subspace U, V):
  LIBERO-90 adapters: shared 0.794 | writes INTO shared outputs 0.002 | reads shared inputs, writes other outputs 0.203
  anchored new:       shared 0.884 | writes INTO shared outputs 0.039 | reads shared inputs, writes other outputs 0.075
  scratch new:        shared 0.001 | writes INTO shared outputs 0.065 | reads shared inputs 0.009 | fully outside 0.925
LIBERO-90 tasks never write into the shared output directions; new adapters do. Hypothesis: that component is what damages
the library when a new task is weighted up. Fix: keep only (I - U U^T) D_p for a privileged task = its update with the
shared output directions removed (equals (I - UU^T) r_p; for LIBERO-90-like adapters it is ~the full residual).

PRE-REGISTERED (10-17, before any rollout):
  model      W = base + S16 + w * sum_{p in P} (I - U U^T) D_p + (1/k) * sum_{i not in P} r_i
  everything else = EXP-120 (P = first 2 new tasks; w = 1, 0.25; scratch + anchored; t = 8; 10 eps, states 0..9)
  readout    privileged / LIBERO-90 / other new retention vs EXP-120 (same w, paired) and vs ours_lib
  predictions anchored: LIBERO-90 loss shrinks clearly vs EXP-120 while the privileged tasks stay protected (their useful part
             is the "reads shared inputs" component, kept). Scratch: little change (the write-into-shared part is 6.5% of a
             mostly-outside update; EXP-120's collapse is unlikely to come from it alone).
Env: E121_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp120_privileged as P
import exp116_continual as C
import exp113_controls as X
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U

class WBuilder(P.PBuilder):
    def _delta(self, m, p, members, s_):
        if m != "wpriv": return super()._delta(m, p, members, s_)
        F = {i: self.fac(i) for i in members}
        if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F.values()): return None
        k = len(members); rest = [i for i in members if i not in p["P"]]
        Us = U.USIG[s_].to(E.DEV).float(); Un = Us / torch.linalg.norm(Us, dim=0).clamp(min=1e-12)
        Bs = [F[i][s_][0].to(E.DEV).float() / k for i in rest]; As = [F[i][s_][1].to(E.DEV).float() for i in rest]
        D = MM.ours(Bs, As, 1.0, Us, U.VS[s_].to(E.DEV).float())
        for i in p["P"]:
            Bp, Ap = F[i][s_][0].to(E.DEV).float(), F[i][s_][1].to(E.DEV).float()
            D = D + p["w"] * ((Bp - Un @ (Un.T @ Bp)) @ Ap)
        return D

def items():
    return [(tg.replace("c120", "c121").replace("-priv_", "-wpriv_"), s, p, mem, ev) for tg, s, p, mem, ev in P.items()]

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[3]), "test", it[1], it[4]) for sc in scs)]
    print(f"[e121] {len(todo)} merged models", flush=True)
    if not todo: return
    B = WBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, p, mem, ev in rd:
            out = E.PRE / f"e121_{tg}"; t1 = time.time(); B.build("wpriv", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e121] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e121_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e121] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    L = ["=== EXP-121 write-protected privileged tasks ((I-UU^T) D_p at weight w) vs EXP-120 (two-sided residual) and ours_lib, t = 8 ==="]
    for cd, (pref, seqf) in P.COND.items():
        ref = [P.groups(f"{pref}s{s}t{P.T}-ours_lib", s, seqf) for s in range(C.NSEQ)]
        for w in P.WS:
            arm = [P.groups(f"c121{cd}s{s}t{P.T}-wpriv_w{w:g}", s, seqf) for s in range(C.NSEQ)]
            e120 = [P.groups(f"c120{cd}s{s}t{P.T}-priv_w{w:g}", s, seqf) for s in range(C.NSEQ)]
            if any(a is None for a in arm + ref + e120): L.append(f"{cd} w={w:g}: incomplete"); continue
            L.append(f"{'scratch' if cd == 'z' else 'anchored'}, w = {w:g}:")
            for g in ("priv", "old", "other new"):
                x = {k_: v for a in arm for k_, v in a[g].items()}
                for nm, R_ in (("EXP-120", e120), ("ours_lib", ref)):
                    y = {k_: v for a in R_ for k_, v in a[g].items()}; ks = [q for q in x if q in y]
                    d = np.array([100 * (x[q] - y[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                    L.append(f"  {g:9s}: {100*np.mean([x[q] for q in ks]):5.1f}% vs {nm:8s} {100*np.mean([y[q] for q in ks]):5.1f}%  "
                             f"({d.mean():+6.1f}pp, better {int((d>0).sum())} / worse {int((d<0).sum())}, p={p_:.4f}, n={len(ks)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp121_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E121_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e121] {st} DONE", flush=True)
