#!/usr/bin/env python3
"""EXP-123 — new skills TRAINED write-protected, then privileged (user OK 10-17).

EXP-117/119/120/121: every weight-space knob trades LIBERO-90 retention for new-task retention on one frontier; the part of a
new adapter that writes into the shared output directions both hurts the library and carries much of the new task, and
removing it after training removes the protection (EXP-121, w=0.25). LIBERO-90 adapters never write into the shared outputs:
their task-specific part reads the 16 shared input directions and writes elsewhere (energy 0.203 vs 0.002).
Train new adapters in exactly that form: S16 init, A frozen at V^T, B's shared-output component held at U*Sigma
(`train_lora.py --init s16wp`) -> D = S16 + B_perp V^T, the task must be learned without writing into the shared outputs.

PRE-REGISTERED (10-17, before any training):
  adapters   the 13 unique privileged tasks of sequences 0-7 (first 2 new tasks each), 1,500 steps, batch 8, as EXP-37
  single     2 x 10 episodes on states 0..9 per adapter (records e123_base_*)
  model      EXP-120's privileged merge with the privileged tasks' anchored adapters replaced by their write-protected ones:
             W = S16 + w * sum_p r_p + (1/k) * sum_others r_i   (r_p = B_perp V^T exactly); other new tasks stay anchored
  arms       w = 1 and 0.25; sequences 0-7; t = 8; 10 episodes, states 0..9
  readout    privileged / LIBERO-90 / other new retention, paired vs ours_lib (anchored) and vs EXP-120/122 privileged
             (anchored adapters, same w) over 16 privileged, 64 LIBERO-90, 48 other new instances
  predictions write-protected training moves OFF the frontier: privileged protected (>= anchored-privileged at the same w
             minus a few pp) while LIBERO-90 stays near ours_lib (loss < 5pp at w = 0.25). Risk: the constraint lowers
             single-task success (reported).
Env: E123_STAGE base|run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp120_privileged as P
import exp118_anchored as A
import exp116_continual as C
import exp113_controls as X
import exp112_bench as E
import unified_pool_driver as U
SEQS, WS = range(0, 8), (1.0, 0.25)
ADIR = REPO / "results/analysis/new_loras_s16wp"
NA = X.NF + X.NE                                                     # first anchored index
WPX, WPI = [], {}                                                    # write-protected entries; anchored index -> wp index
_priv = sorted({i for s in SEQS for i in A.sequence(s)[1][:P.NP]})
for i in _priv:
    e = E.LIB[i]; name = str(e["key"]).split(":", 1)[1]; path = ADIR / name / "adapter_model.safetensors"
    U.EXTPATH["np16wp:" + name] = str(path)
    WPI[i] = len(E.LIB) + len(WPX); WPX.append(dict(e, key="np16wp:" + name, tid="w" + e["tid"][1:], path=str(path)))
E.LIB.extend(WPX)

def single():
    s = X.single()
    for i, e in WPI.items():
        rs = [E.rec(f"e123_base_{E.LIB[e]['tid']}_r{r}", E.LIB[e]["scene"]) for r in (1, 2)]; rs = [x for x in rs if x]
        if rs: s[e] = np.mean([x["success_rate"] for x in rs])
    return s

def sequence(s):
    old, new = A.sequence(s); return old, [WPI.get(i, i) if q < P.NP else i for q, i in enumerate(new)]

def items():
    out = []
    for s in SEQS:
        old, new = sequence(s); mem = old + new[:P.T]; ev = sorted(old[:C.NEVAL] + new[:P.T])
        for w in WS: out.append((f"c123s{s}t8-wppriv_w{w:g}", s, {"w": w, "P": new[:P.NP]}, mem, ev))
    return out

def base():
    J = [{"name": f"e123_base_{e['tid']}_r{r}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [e["path"]],
          "eval_scenes": [e["scene"]], "suite": e["suite"], "norm_stats": U.FLEET_NORM}
         for r in (1, 2) for e in WPX if E.rec(f"e123_base_{e['tid']}_r{r}", e["scene"]) is None]
    missing = [e["path"] for e in WPX if not pathlib.Path(e["path"]).exists()]
    if missing: raise SystemExit(f"[e123] {len(missing)} adapters not trained: {missing[:2]}")
    print(f"[e123] base: {len(J)} single-task jobs", flush=True)
    if J: subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(J, 10, 0, "e123base"))], cwd=REPO)

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[3]), "test", it[1], it[4]) for sc in scs)]
    print(f"[e123] {len(todo)} merged models", flush=True)
    if not todo: return
    B = P.PBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, p, mem, ev in rd:
            out = E.PRE / f"e123_{tg}"; t1 = time.time(); B.build("priv", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e123] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e123_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e123] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def groups(tg, s, seqf, sg):
    """retention per task keyed by (sequence, position): old by task index, new by arrival position (same task across
    anchored / write-protected adapters)"""
    old, new = seqf(s); ev = sorted(old[:C.NEVAL] + new[:P.T])
    sc = E.member_success(tg, len(old) + P.T, "test", s, ev)
    if len(sc) < len(ev): return None
    r = {i: sc[i] / sg[i] for i in ev if sg.get(i, 0) > 0}
    return {"old": {(s, i): v for i, v in r.items() if i < X.NF},
            "priv": {(s, q): r[i] for q, i in enumerate(new[:P.NP]) if i in r},
            "other new": {(s, q): r[i] for q, i in enumerate(new[:P.T]) if q >= P.NP and i in r}}

def report():
    from scipy import stats
    sg = single()
    L = ["=== EXP-123 write-protected TRAINED privileged tasks (anchored others), t = 8, sequences 0-7 ===",
         "single-task success, write-protected vs anchored adapters of the same 13 tasks: " +
         (f"{100*np.mean([sg[e] for e in WPI.values() if e in sg]):.1f}% vs {100*np.mean([sg[i] for i in WPI if i in sg]):.1f}% "
          f"(n={sum(e in sg for e in WPI.values())})")]
    refs = {"ours_lib": lambda s, w: f"c118s{s}t8-ours_lib",
            "anchored privileged": lambda s, w: f"c120as{s}t8-priv_w{w:g}",
            "post-hoc write-protected": lambda s, w: f"c121as{s}t8-wpriv_w{w:g}"}
    for w in WS:
        arm = [groups(f"c123s{s}t8-wppriv_w{w:g}", s, sequence, sg) for s in SEQS]
        if any(a is None for a in arm): L.append(f"w={w:g}: incomplete"); continue
        L.append(f"w = {w:g}:")
        for nm, f in refs.items():
            ref = [groups(f(s, w), s, A.sequence, sg) for s in SEQS]
            if any(r is None for r in ref): L.append(f"  vs {nm}: reference incomplete"); continue
            for g in ("priv", "old", "other new"):
                x = {k_: v for a in arm for k_, v in a[g].items()}; y = {k_: v for a in ref for k_, v in a[g].items()}; ks = [q for q in x if q in y]
                d = np.array([100 * (x[q] - y[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                L.append(f"  {g:9s}: {100*np.mean([x[q] for q in ks]):5.1f}% vs {nm:24s} {100*np.mean([y[q] for q in ks]):5.1f}%  "
                         f"({d.mean():+6.1f}pp, better {int((d>0).sum())} / worse {int((d<0).sum())}, p={p_:.4f}, n={len(ks)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp123_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E123_STAGE", "run"); U._clear_orphaned_builds()
    {"base": base, "run": run, "report": report}[st]()
    print(f"[e123] {st} DONE", flush=True)
