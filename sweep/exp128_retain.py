#!/usr/bin/env python3
"""EXP-128 — RETAIN-style recursive interpolation as a continual-merging baseline (user OK 10-19).

RETAIN (Yadav et al., ICLR'26, arXiv 2512.08333) merges sequentially: theta_n = (1 - a) theta_{n-1} + a theta_ft,n (task n
fine-tuned from the previous merge; full fine-tunes; a ~ 0.5 typical, tuned on a held-out OOD scene). Training-free analog
with our independently trained adapters: the model holding the 16 LIBERO-90 tasks starts as their average; each arriving
task applies Delta <- (1 - a) Delta + a D_new. Final coefficients: old task (1/16)(1 - a)^t; new task j (arrival j = 1..t)
a (1 - a)^(t - j) -> order-dependent, older contributions decay geometrically (a = 1/k recovers plain averaging).

PRE-REGISTERED (10-19, before any rollout): EXP-124 setting (sequences 0-3, every new task write-protected, t = 4, 8, same
evaluated tasks, 10 eps, states 0..9), a = 0.5 (RETAIN's typical value; no tuning); 8 models, 1,120 episodes. Readout vs
ours (w = 0.1, 0.25), OPCM, averaging (EXP-124/125 records), paired at t = 8. Prediction: LIBERO-90 retention far below ours
(old residuals decay as 0.5^t; only the shared part carried by the write-protected adapters survives) while the most
recent new tasks are learned well and the earliest are forgotten.
Env: E128_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp124_wpcontinual as W
import exp120_privileged as P
import exp116_continual as C
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U
ALPHA = 0.5
SCRATCH = os.environ.get("E128_ADAPTERS") == "scratch"               # 10-19 add-on: training-free case (EXP-116 adapters)
SEQF = C.sequence if SCRATCH else W.sequence
PFX = "c128z" if SCRATCH else "c128"

class RBuilder(P.PBuilder):
    def _delta(self, m, p, members, s_):
        if m != "retain": return super()._delta(m, p, members, s_)
        F = [self.fac(i) for i in members]
        if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F): return None
        n_old, t, a = p["n_old"], len(members) - p["n_old"], p["a"]
        w = [(1.0 / n_old) * (1 - a) ** t] * n_old + [a * (1 - a) ** (t - j) for j in range(1, t + 1)]
        return MM.ta([wi * f[s_][0].to(E.DEV) for wi, f in zip(w, F)], [f[s_][1].to(E.DEV) for f in F], 1.0)

def items():
    out = []
    for s in W.SEQS:
        old, new = SEQF(s)
        for t in W.TS:
            mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t])         # arrival order kept in mem
            out.append((f"{PFX}s{s}t{t}-retain_a{ALPHA:g}", s, t, {"a": ALPHA, "n_old": len(old)}, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[4]), "test", it[1], it[5]) for sc in scs)]
    print(f"[e128] {len(todo)} merged models", flush=True)
    if not todo: return
    B = RBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, p, mem, ev in rd:
            out = E.PRE / f"e128_{tg}"; t1 = time.time(); B.build("retain", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e128] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e128_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e128] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    sg = W.single()
    arms = ([("RETAIN a=0.5", "c128zs{s}t{t}-retain_a0.5"), ("ours w=0.25", "c116s{s}t{t}-ours_lib"), ("ours w=0.1", "c116s{s}t{t}-ours_lib"),
             ("OPCM", "c116s{s}t{t}-opcm"), ("averaging", "c116s{s}t{t}-avg")] if SCRATCH else
            [("RETAIN a=0.5", "c128s{s}t{t}-retain_a0.5"), ("ours w=0.25", "c124s{s}t{t}-wp_w0.25"), ("ours w=0.1", "c124s{s}t{t}-wp_w0.1"),
             ("OPCM", "c125s{s}t{t}-opcm"), ("averaging", "c125s{s}t{t}-avg")])
    L = ["=== EXP-128 RETAIN-style recursive interpolation (a=0.5) vs ours / OPCM / averaging, write-protected adapters, sequences 0-3 ==="]
    for t in W.TS:
        G = {}
        for nm, f in arms:
            g = [W.groups(f.format(s=s, t=t), s, t, SEQF, sg) for s in W.SEQS]
            if any(x is None for x in g): L.append(f"t={t} {nm:13s} incomplete"); continue
            G[nm] = g
            L.append(f"t={t} {nm:13s} old {100*np.mean([v for x in g for v in x['old'].values()]):5.1f}%  new {100*np.mean([v for x in g for v in x['new'].values()]):5.1f}%")
        if "RETAIN a=0.5" in G:                                          # new-task retention by arrival position (recency)
            pos = {q: [x["new"][(s, q)] for s, x in zip(W.SEQS, G["RETAIN a=0.5"]) if (s, q) in x["new"]] for q in range(t)}
            L.append(f"t={t} RETAIN new-task retention by arrival position 1..{t}: " + " ".join(f"{100*np.mean(v):.0f}" for q, v in pos.items()))
        if t == 8 and "RETAIN a=0.5" in G and "ours w=0.25" in G:
            for nm in ("ours w=0.25", "ours w=0.1"):
                line = f"  {nm} vs RETAIN"
                for k_ in ("old", "new"):
                    x_ = {q: v for z in G[nm] for q, v in z[k_].items()}; y_ = {q: v for z in G["RETAIN a=0.5"] for q, v in z[k_].items()}; ks = [q for q in x_ if q in y_]
                    d = np.array([100 * (x_[q] - y_[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                    line += f" | {k_} {d.mean():+6.1f}pp ({int((d>0).sum())}/{int((d<0).sum())}, p={p_:.4f})"
                L.append(line)
    if SCRATCH: L[0] = L[0].replace("write-protected adapters", "SCRATCH adapters (EXP-116); 'ours w=0.25/0.1' rows = ours_lib (1/k)")
    txt = "\n".join(L); print(txt); (REPO / f"results/analysis/exp128{'z' if SCRATCH else ''}_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E128_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e128] {st} DONE", flush=True)
