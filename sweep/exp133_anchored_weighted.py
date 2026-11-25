#!/usr/bin/env python3
"""EXP-133 — write-protected vs plain fine-tuning from the shared part, in the headline setting (user question 10-20).

The headline (EXP-124/127) weights ALL new skills up (w = 0.1, 0.25) with write-protected adapters; the same merge with the
anchored adapters (S16 init, A and B trained freely — "fine-tune from the shared part") was never run. EXP-120/126 suggest
write-protection wins at strong weighting (w=1) but may tie at moderate weights.

PRE-REGISTERED (10-20, before any rollout): EXP-118 sequences 0-3 (anchored adapters for every new skill), t = 4, 8, same
evaluated tasks, 10 eps, states 0..9; arms ours w = 0.1, 0.25 (W = S16 + w * sum_new r + (1/k) * sum_old r), exactly the
EXP-124 rule; 16 models. Readout: vs EXP-124 (write-protected, same orders / weights / task positions), paired.
Prediction: at w = 0.1 the two are equal (n.s.); at w = 0.25 write-protected keeps more LIBERO-90 (>= +5pp) at similar
new-skill retention.
Env: E133_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp124_wpcontinual as W
import exp120_privileged as P
import exp118_anchored as A
import exp116_continual as C
import exp112_bench as E
import unified_pool_driver as U
WS = (0.1, 0.25)

def items():
    out = []
    for s in W.SEQS:
        old, new = A.sequence(s)
        for t in W.TS:
            mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t])
            for w in WS: out.append((f"c133s{s}t{t}-anch_w{w:g}", s, t, {"w": w, "P": new[:t]}, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[4]), "test", it[1], it[5]) for sc in scs)]
    todo.sort(key=lambda it: -it[2])                                    # 10-21: t = 8 first (meeting at 14:00)
    print(f"[e133] {len(todo)} merged models", flush=True)
    if not todo: return
    B = P.PBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, p, mem, ev in rd:
            out = E.PRE / f"e133_{tg}"; t1 = time.time(); B.build("priv", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e133] built {tg} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e133_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e133] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    sg = W.single()
    L = ["=== EXP-133 anchored (fine-tuned from the shared part) vs write-protected adapters, same merge, sequences 0-3 ==="]
    for t in W.TS:
        for w in WS:
            ga = [W.groups(f"c133s{s}t{t}-anch_w{w:g}", s, t, A.sequence, sg) for s in W.SEQS]
            gw = [W.groups(f"c124s{s}t{t}-wp_w{w:g}", s, t, W.sequence, sg) for s in W.SEQS]
            if any(x is None for x in ga + gw): L.append(f"t={t} w={w:g}: incomplete"); continue
            line = f"t={t} w={w:g}"
            for k_ in ("old", "new"):
                a = {q: v for z in ga for q, v in z[k_].items()}; b = {q: v for z in gw for q, v in z[k_].items()}; ks = [q for q in a if q in b]
                d = np.array([100 * (b[q] - a[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                line += f" | {k_}: anchored {100*np.mean([a[q] for q in ks]):5.1f}% write-protected {100*np.mean([b[q] for q in ks]):5.1f}% (WP - anch {d.mean():+5.1f}pp, p={p_:.3f})"
            L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp133_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E133_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e133] {st} DONE", flush=True)
