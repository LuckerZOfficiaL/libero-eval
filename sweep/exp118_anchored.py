#!/usr/bin/env python3
"""EXP-118 — continual merging with new skills TRAINED FROM THE SHARED PART (follow-up to EXP-116/117, 10-16).

EXP-116: with new tasks trained from scratch (zero-init LoRA), our merge keeps LIBERO-90 skills flat but no method learns
the new tasks (<=21% retention after 8 additions). EXP-117: weighting new tasks more (c/k, c = 2/4/8) does not help them
and destroys the old ones -> plasticity is not a dilution problem a re-weighting can fix. EXP-113 condition D showed that
external adapters trained starting from the library's shared part (S16-init, 30 adapters, same tasks) merge like library
tasks (averaging 58 -> 76% in EXP-112's mixed subsets). Question: is "train new skills from the shared part, then merge"
the plasticity route in continual merging, and does our merge still matter then?

PRE-REGISTERED (10-16, before any rollout):
  sequences  EXP-116's 4 sequences and arrival orders, each new task replaced by its S16-init adapter (same task)
  checkpoints t = 4, 8 additions (t = 0 has no new task = EXP-116, reused); evaluated: the same 8 old tasks + all new so far;
             10 episodes, states 0..9
  methods    EXP-116's rules unchanged: avg; ta 1.33/k; ties, dare_ties norm-matched x1.33; cart 0.25/0.08; maxabs x1.33;
             opcm alpha 0.5 (arrival order, norm rescaling); ours_lib (LIBERO-90 shared part, alpha 1/k)
             (ours_all dropped: answered in EXP-116)
  metrics    old / new retention (new retention vs the S16-init adapters' own single-task success), compared with EXP-116
  predictions new-task retention rises well above EXP-116's for the linear methods; averaging forgets far less than in
             EXP-116 (new adapters carry the shared part, no dilution); ours ~ averaging (as in EXP-113 D).
Env: E118_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp116_continual as C
import exp113_controls as X
import exp112_bench as E
import unified_pool_driver as U
METHODS = [x for x in C.METHODS if x[0] != "ours_all"]
TS = (4, 8)

def sequence(s):
    old, new = C.sequence(s)
    return old, [i + X.NE for i in new]                                  # S16-init adapter of the same task

def items():
    out = []
    for s in range(C.NSEQ):
        old, new = sequence(s)
        for t in TS:
            mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t]); k = len(mem)
            for m, p, c in METHODS:
                if m == "ta": p = {"lam": 1.33 / k}
                out.append((f"c118s{s}t{t}-{m}", s, t, m, p, c, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[7]) for sc in scs)]
    print(f"[e118] {len(todo)} merged models", flush=True)
    if not todo: return
    B = C.CBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, m, p, c, mem, ev in rd:
            out = E.PRE / f"e118_{tg}"; t1 = time.time(); B.build(m, p, c, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e118] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e118_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e118] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def inst(pref, seqf, s, t, m, which):
    old, new = seqf(s); k = len(old) + t; ev = sorted(old[:C.NEVAL] + new[:t]); sg = X.single()
    sc = E.member_success(f"{pref}s{s}t{t}-{m}", k, "test", s, ev)
    if len(sc) < len(ev): return None
    return {(s, ev.index(i) if which == "new" else i): sc[i] / sg[i] for i in ev
            if (which == "old") == (i < X.NF) and sg.get(i, 0) > 0}

def report():
    from scipy import stats
    def pool(pref, seqf, t, m, which):
        o = {}
        for s in range(C.NSEQ):
            a = inst(pref, seqf, s, t, m, which)
            if a is None: return None
            o.update(a)
        return o
    L = ["=== EXP-118 continual merging, new tasks trained FROM the shared part (S16-init) vs from scratch (EXP-116) ===",
         f"{'t':>2} {'method':9s} {'old: scratch':>12s} {'old: anchored':>13s} {'new: scratch':>12s} {'new: anchored':>13s}"]
    R = {}
    for t in TS:
        for m, _, _ in METHODS:
            v = [pool("c116", C.sequence, t, m, "old"), pool("c118", sequence, t, m, "old"),
                 pool("c116", C.sequence, t, m, "new"), pool("c118", sequence, t, m, "new")]
            R[(t, m)] = v
            f = lambda d: f"{100*np.mean(list(d.values())):11.1f}%" if d else "  incomplete"
            L.append(f"{t:>2} {m:9s} {f(v[0])} {f(v[1]):>13s} {f(v[2])} {f(v[3]):>13s}")
    L.append("t = 8, anchored: ours_lib vs each method (paired over task instances)")
    for which, j in (("old", 1), ("new", 3)):
        ref = R[(8, "ours_lib")][j]
        if not ref: continue
        for m, _, _ in METHODS:
            if m == "ours_lib" or not R[(8, m)][j]: continue
            o = R[(8, m)][j]; ks = [q for q in ref if q in o]; d = np.array([100 * (ref[q] - o[q]) for q in ks])
            p = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
            L.append(f"  {which}: ours - {m:9s} {d.mean():+6.1f}pp  better {int((d>0).sum())} / worse {int((d<0).sum())}  p={p:.4f}")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp118_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E118_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e118] {st} DONE", flush=True)
