#!/usr/bin/env python3
"""EXP-122 — replication of the privileged-tasks frontier on 4 NEW sequences (anchored adapters), 10-17.

EXP-120/121 rest on 8 privileged task instances (2 per sequence x 4 sequences): w=0.25 protection p=0.06, write-protected
w=1 p=0.04. Same design on sequences 4..7 (seeded exactly like EXP-116, rng 11600+s), anchored adapters, t = 8:
  arms       ours_lib (reference) · priv w=1 · priv w=0.25 · write-protected priv w=1   (definitions: EXP-118/120/121 drivers)
  readout    pooled over all 8 sequences (n = 16 privileged, 64 LIBERO-90, 48 other new instances), paired vs ours_lib
  prediction the EXP-120/121 pattern replicates on the new sequences (no re-tuning; same fixed rules)
Env: E122_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp121_writeprotect as W
import exp120_privileged as P
import exp118_anchored as A
import exp116_continual as C
import exp112_bench as E
import unified_pool_driver as U
SEQS, ALLSEQ = range(4, 8), range(0, 8)
ARMS = [("ours_lib", None, "c118s{s}t8-ours_lib"), ("priv", 1.0, "c120as{s}t8-priv_w1"), ("priv", 0.25, "c120as{s}t8-priv_w0.25"),
        ("wpriv", 1.0, "c121as{s}t8-wpriv_w1")]

def items():
    out = []
    for s in SEQS:
        old, new = A.sequence(s); mem = old + new[:P.T]; ev = sorted(old[:C.NEVAL] + new[:P.T])
        for m, w, tg in ARMS:
            p = {} if m == "ours_lib" else {"w": w, "P": new[:P.NP]}
            out.append((tg.format(s=s), s, m, p, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[4]), "test", it[1], it[5]) for sc in scs)]
    print(f"[e122] {len(todo)} merged models", flush=True)
    if not todo: return
    B = W.WBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, m, p, mem, ev in rd:
            out = E.PRE / f"e122_{tg}"; t1 = time.time(); B.build(m, p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e122] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e122_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e122] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    L = ["=== EXP-122 privileged-tasks frontier, anchored, t = 8: new sequences 4-7 alone, and pooled 0-7 ==="]
    for lab, seqs in (("new sequences 4-7", SEQS), ("pooled 0-7", ALLSEQ)):
        ref = [P.groups(ARMS[0][2].format(s=s), s, A.sequence) for s in seqs]
        if any(r is None for r in ref): L.append(f"{lab}: reference incomplete"); continue
        L.append(f"{lab}:")
        for m, w, tg in ARMS[1:]:
            arm = [P.groups(tg.format(s=s), s, A.sequence) for s in seqs]
            if any(a is None for a in arm): L.append(f"  {tg[6:]}: incomplete"); continue
            for g in ("priv", "old", "other new"):
                x = {k_: v for a in arm for k_, v in a[g].items()}; y = {k_: v for a in ref for k_, v in a[g].items()}; ks = [q for q in x if q in y]
                d = np.array([100 * (x[q] - y[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                L.append(f"  {m:5s} w={w:<4g} {g:9s}: {100*np.mean([x[q] for q in ks]):5.1f}% vs ours_lib {100*np.mean([y[q] for q in ks]):5.1f}%  "
                         f"({d.mean():+6.1f}pp, better {int((d>0).sum())} / worse {int((d<0).sum())}, p={p_:.4f}, n={len(ks)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp122_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E122_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e122] {st} DONE", flush=True)
