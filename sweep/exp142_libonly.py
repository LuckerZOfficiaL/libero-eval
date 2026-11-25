#!/usr/bin/env python3
"""EXP-142 — does the library alone already solve the externals that survive in ours? (10-25, after EXP-140)

Per-skill EXP-134 (scratch, ours): external retention is bimodal and fixed on arrival — 6 of 32 (order, external)
instances keep 30-100% from the step they are added to t=8, all others are at 0% from the start (BWT ~ 0). Each external
enters at weight 1/k (~1/17), so the survivors may owe nothing to their own LoRA. Model per order s (EXP-116 sequences,
orders 0-3): the library-only merge (t = 0: S + (1/16) sum of library residuals), evaluated on the order's 8 external
scenes, 10 eps, states 0..9. Compare with ours at the step each external is added (EXP-134).
PRE-REGISTERED (10-25, before any rollout): if the library alone reaches >= 0.8 x ours' retention on the survivors, the
externals' own LoRAs contribute ~nothing in ours -> the method must give externals far more weight on their own inputs.
4 models, 32 evaluations. Env: E142_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp140_softgate as P
E, C, U, X = P.E, P.C, P.U, P.X
SEQS = (0, 1, 2, 3)

def items():
    out = []
    for s in SEQS:
        old, new = C.sequence(s); out.append((f"c142s{s}-lib", s, "plain", None, old, [], sorted(new)))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[6]) for sc in scs)]
    print(f"[e142] {len(todo)} models, {sum(len(x[6]) for x in todo)} evaluations", flush=True)
    if not todo: return
    B = P.GBuilder(); cur, prev = [], None
    for r, it in enumerate(todo):
        out = E.PRE / f"e142_{it[0]}"; t1 = time.time(); B.build_arm(it, out); js = []
        for name, su, scenes, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[6]):
            if all(E.rec(name, sc) is not None for sc in scenes): continue
            js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                       "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
        print(f"[e142] built {it[0]} in {time.time()-t1:.0f}s", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(js, 10, 0, f"e142_{r}"))], cwd=REPO)
        shutil.rmtree(out, ignore_errors=True)
        print(f"[e142] round {r + 1}/{len(todo)} done", flush=True)

def report():
    sg = X.single(); L = ["=== EXP-142 library alone on the externals vs ours at the step each external is added (retention) ==="]
    a, b = [], []
    for tg, s, kind, w, old, _, ev in items():
        new = C.sequence(s)[1]; lib = E.member_success(tg, len(ev), "test", s, ev)
        for t, e in enumerate(new, 1):
            mem = old + new[:t]; o = E.member_success(f"c134zs{s}t{t}-ours", len(mem), "test", s, sorted(mem))
            if e not in lib or e not in o or not sg.get(e): continue
            a.append(lib[e] / sg[e]); b.append(o[e] / sg[e])
            L.append(f"order {s} t={t} {E.LIB[e]['tid']:9s} own {100*sg[e]:3.0f}%  library alone {100*a[-1]:4.0f}%  ours {100*b[-1]:4.0f}%")
    a, b = np.array(a), np.array(b); surv = b >= 0.25
    if len(a):
        L.append(f"all n={len(a)}: library alone {100*a.mean():.1f}% vs ours {100*b.mean():.1f}%")
        if surv.any(): L.append(f"survivors in ours (>= 25%, n={surv.sum()}): library alone {100*a[surv].mean():.1f}% vs ours {100*b[surv].mean():.1f}%")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp142_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st_ = os.environ.get("E142_STAGE", "run"); U._clear_orphaned_builds()
    run() if st_ == "run" else report()
    print(f"[e142] {st_} DONE", flush=True)
