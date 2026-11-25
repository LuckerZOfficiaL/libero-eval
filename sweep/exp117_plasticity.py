#!/usr/bin/env python3
"""EXP-117 — PLASTICITY knob for training-free continual merging (follow-up to EXP-116, 10-15).

EXP-116 finding: our merge (LIBERO-90 shared part at full strength + residuals at 1/k) keeps LIBERO-90 skills almost
flat as 8 external tasks arrive (old retention 82 -> 80%; avg 86 -> 63%), but NO method learns the new tasks
(new-task retention 9-21% after 8 additions). Mechanism: external adapters have <1% energy in the shared part, so a new
task lives entirely in its residual, which is diluted to 1/k (k = 20..24).
Question: if new tasks get a larger weight c/k (old tasks stay at 1/k), does the shared part let our merge buy
plasticity at a smaller stability cost than plain averaging given the same boost?

PRE-REGISTERED (10-15, before any rollout):
  sequences, checkpoints, evaluated tasks, episodes, states: identical to EXP-116 (4 sequences of 16 LIBERO-90 + 8 external
             zero-init tasks; t = 4, 8 additions; a fixed 8 of the 16 old + every new task so far; 10 episodes, states 0..9)
  arms       ours_asym(c): W = base + S16(LIBERO-90 library) + sum_i w_i r_i,   w_i = 1/k (old), c/k (new)
             avg_asym(c) : W = base + sum_i w_i D_i,                           same weights (the matched control)
             c in {2, 4, 8}; c = 1 is EXP-116 (ours_lib, avg) and is reused, not rerun
  readout    per c and t: old retention (stability), new retention (plasticity); trade-off curve old-vs-new per method
  primary    t = 8: ours_asym(c) - avg_asym(c) on old retention and on new retention, paired over task instances
             (Wilcoxon), for each c
  predictions new retention rises with c for both arms; old retention falls with c for both, less for ours (shared part
             anchors the old tasks); ours' curve dominates avg's. Not a tuning run: all c are reported.
Env: E117_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp116_continual as C
import exp113_controls as X
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U
CS, TS = (2, 4, 8), (4, 8)
ARMS = ("ours_asym", "avg_asym")

class ABuilder(C.CBuilder):
    def _delta(self, m, p, members, s_):
        if m not in ARMS: return super()._delta(m, p, members, s_)
        F = [self.fac(i) for i in members]
        if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F): return None
        k = len(members); w = [(p["c"] if i >= X.NF else 1.0) / k for i in members]
        Bs = [wi * f[s_][0].to(E.DEV) for wi, f in zip(w, F)]; As = [f[s_][1].to(E.DEV) for f in F]
        if m == "avg_asym": return MM.ta(Bs, As, 1.0)
        return MM.ours(Bs, As, 1.0, U.USIG[s_].to(E.DEV), U.VS[s_].to(E.DEV))

def tg(s, t, m, c):
    if c == 1: return f"c116s{s}t{t}-{'ours_lib' if m == 'ours_asym' else 'avg'}"   # EXP-116 records
    return f"c117s{s}t{t}-{m}_c{c}"

def items():
    out = []
    for c in CS:
        for s in range(C.NSEQ):
            old, new = C.sequence(s)
            for t in TS:
                mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t])
                for m in ARMS: out.append((tg(s, t, m, c), s, t, m, {"c": c}, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[5]), "test", it[1], it[6]) for sc in scs)]
    print(f"[e117] {len(todo)} merged models", flush=True)
    if not todo: return
    B = ABuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for name_, s, t, m, p, mem, ev in rd:
            out = E.PRE / f"e117_{name_}"; t1 = time.time(); B.build(m, p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(name_, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e117] built {name_} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e117_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e117] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def inst(s, t, m, c, which):
    old, new = C.sequence(s); k = len(old) + t; ev = sorted(old[:C.NEVAL] + new[:t]); sg = X.single()
    sc = E.member_success(tg(s, t, m, c), k, "test", s, ev)
    if len(sc) < len(ev): return None
    return {(s, i): sc[i] / sg[i] for i in ev if (which == "old") == (i < X.NF) and sg.get(i, 0) > 0}

def report():
    from scipy import stats
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    L = ["=== EXP-117 plasticity knob: new tasks weighted c/k, old 1/k (c = 1 is EXP-116); retention, 4 sequences ===",
         f"{'t':>2} {'c':>2} {'arm':10s} {'old ret':>8s} {'new ret':>8s} {'mean':>7s}"]
    pts = {}
    for t in TS:
        for c in (1,) + CS:
            for m in ARMS:
                o, n = {}, {}
                for s in range(C.NSEQ):
                    a, b = inst(s, t, m, c, "old"), inst(s, t, m, c, "new")
                    if a is None or b is None: continue
                    o.update(a); n.update(b)
                if not o: L.append(f"{t:>2} {c:>2} {m:10s} incomplete"); continue
                po, pn = 100 * np.mean(list(o.values())), 100 * np.mean(list(n.values())); pts[(t, c, m)] = (po, pn, o, n)
                L.append(f"{t:>2} {c:>2} {m:10s} {po:7.1f}% {pn:7.1f}% {(po + pn) / 2:6.1f}%")
    L.append("primary (t = 8): ours_asym - avg_asym, paired over task instances")
    for c in (1,) + CS:
        if (8, c, "ours_asym") not in pts or (8, c, "avg_asym") not in pts: continue
        for j, nm in ((2, "old"), (3, "new")):
            A, Bv = pts[(8, c, "ours_asym")][j], pts[(8, c, "avg_asym")][j]; ks = [q for q in A if q in Bv]
            d = np.array([100 * (A[q] - Bv[q]) for q in ks]); p = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
            L.append(f"  c={c} {nm}: {d.mean():+6.1f}pp  better {int((d>0).sum())} / worse {int((d<0).sum())} / tie {int((d==0).sum())}  p={p:.4f}  (n={len(ks)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp117_report.txt").write_text(txt + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), dpi=200, sharey=True)
    for ax, t in zip(axs, TS):
        for m, col, lab in (("ours_asym", "tab:red", "ours (shared part + weighted residuals)"), ("avg_asym", "tab:blue", "weighted averaging")):
            cs = [c for c in (1,) + CS if (t, c, m) in pts]
            if not cs: continue
            x = [pts[(t, c, m)][1] for c in cs]; y = [pts[(t, c, m)][0] for c in cs]
            ax.plot(x, y, "o-", color=col, label=lab)
            for c, xi, yi in zip(cs, x, y): ax.annotate(f"c={c}", (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_title(f"after {t} new tasks (model holds {16 + t})", fontsize=9); ax.set_xlabel("new-task retention (%)", fontsize=8); ax.grid(alpha=0.25)
    axs[0].set_ylabel("LIBERO-90 retention (%)", fontsize=8); axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); (REPO / "results/figs").mkdir(exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(REPO / f"results/figs/exp117_tradeoff.{ext}")

if __name__ == "__main__":
    st = os.environ.get("E117_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e117] {st} DONE", flush=True)
