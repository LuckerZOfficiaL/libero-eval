#!/usr/bin/env python3
"""EXP-124 — continual merging with ALL new skills trained write-protected (follow-up to EXP-123, 10-18).

EXP-123: training a new adapter write-protected (S16 init, A frozen at V^T, no writes into the shared outputs) costs no
single-task success (96.2% vs 95.0%) and moves the privileged-task frontier outward: 2 privileged tasks at w=0.25 are
protected (41% vs 20%) at NO LIBERO-90 cost (77.9% vs 78.3%). Continual question: if every new skill is trained this way,
can all new tasks be weighted up (learned) while LIBERO-90 is kept? Reference points on the same sequences (anchored
adapters): ours_lib 85.3 / 20.2 (old / new retention after 8), OPCM 70.9 / 39.1, ours + OPCM residuals 65.1 / 45.1.

PRE-REGISTERED (10-18, before any training of the 11 extra adapters):
  adapters   write-protected versions of all 22 new tasks of sequences 0-3 (11 from EXP-123 + 11 trained here, same recipe)
  sequences  EXP-116/118 sequences 0-3, every new task replaced by its write-protected adapter; t = 4, 8; same evaluated
             tasks (8 LIBERO-90 + all new so far), 10 episodes, states 0..9; single-task baselines 2 x 10 eps
  arms       W = S16 + w * sum_new r_j + (1/k) * sum_old r_i, w in {1/k (plain), 0.1, 0.25}
  readout    old / new retention; paired vs EXP-118 ours_lib (anchored); means vs OPCM and ours+OPCM (EXP-118/119)
  predictions plain (1/k): ~ anchored ours_lib (new ~20%); w = 0.25: new well above 20% (>= 40%) with LIBERO-90 loss < 5pp
             vs ours_lib -> beats OPCM and ours+OPCM on both axes (off their frontier)
Env: E124_STAGE base|run|report; E113_GPUS / E113_DEV / E113_ROUND / E113_PER_GPU (via exp113).
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
SEQS, TS, WS = range(4), (4, 8), ("1/k", 0.1, 0.25)
ADIR = REPO / "results/analysis/new_loras_s16wp"
WPX, WPI = [], {}
for i in sorted({j for s in SEQS for j in A.sequence(s)[1][:8]}):
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
    old, new = A.sequence(s); return old, [WPI[i] for i in new]

def wtag(w): return "plain" if w == "1/k" else f"w{w:g}"

def items():
    out = []
    for s in SEQS:
        old, new = sequence(s)
        for t in TS:
            mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t]); k = len(mem)
            for w in WS: out.append((f"c124s{s}t{t}-wp_{wtag(w)}", s, t, {"w": 1.0 / k if w == "1/k" else w, "P": new[:t]}, mem, ev))
    return out

def base():
    missing = [e["path"] for e in WPX if not pathlib.Path(e["path"]).exists()]
    if missing: raise SystemExit(f"[e124] {len(missing)} adapters not trained: {missing[:2]}")
    J = [{"name": f"e123_base_{e['tid']}_r{r}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [e["path"]],
          "eval_scenes": [e["scene"]], "suite": e["suite"], "norm_stats": U.FLEET_NORM}
         for r in (1, 2) for e in WPX if E.rec(f"e123_base_{e['tid']}_r{r}", e["scene"]) is None]
    print(f"[e124] base: {len(J)} single-task jobs", flush=True)
    if J: subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(J, 10, 0, "e124base"))], cwd=REPO)

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[4]), "test", it[1], it[5]) for sc in scs)]
    print(f"[e124] {len(todo)} merged models", flush=True)
    if not todo: return
    B = P.PBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, p, mem, ev in rd:
            out = E.PRE / f"e124_{tg}"; t1 = time.time(); B.build("priv", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e124] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e124_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e124] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def groups(tg, s, t, seqf, sg):
    old, new = seqf(s); ev = sorted(old[:C.NEVAL] + new[:t])
    sc = E.member_success(tg, len(old) + t, "test", s, ev)
    if len(sc) < len(ev): return None
    r = {i: sc[i] / sg[i] for i in ev if sg.get(i, 0) > 0}
    return {"old": {(s, i): v for i, v in r.items() if i < X.NF}, "new": {(s, q): r[i] for q, i in enumerate(new[:t]) if i in r}}

def report():
    from scipy import stats
    sg = single()
    L = ["=== EXP-124 continual merging, ALL new tasks write-protected-trained (sequences 0-3), retention old / new ===",
         "single-task success: write-protected %.1f%% vs anchored %.1f%% (same %d tasks)" % (
             100 * np.mean([sg[e] for e in WPI.values() if e in sg]), 100 * np.mean([sg[i] for i in WPI if i in sg]), sum(e in sg for e in WPI.values()))]
    refs = [("anchored ours_lib", "c118s{s}t{t}-ours_lib"), ("anchored OPCM", "c118s{s}t{t}-opcm"), ("anchored ours+OPCM", "c119as{s}t{t}-ours_opcm")]
    for t in TS:
        R = {nm: [groups(f.format(s=s, t=t), s, t, A.sequence, sg) for s in SEQS] for nm, f in refs}
        for nm, rr in R.items():
            if any(x is None for x in rr): L.append(f"t={t} {nm}: incomplete"); continue
            L.append(f"t={t} {nm:22s} old {100*np.mean([v for x in rr for v in x['old'].values()]):5.1f}%  new {100*np.mean([v for x in rr for v in x['new'].values()]):5.1f}%")
        for w in WS:
            arm = [groups(f"c124s{s}t{t}-wp_{wtag(w)}", s, t, sequence, sg) for s in SEQS]
            if any(x is None for x in arm): L.append(f"t={t} wp {wtag(w)}: incomplete"); continue
            line = f"t={t} write-protected {wtag(w):6s} old {100*np.mean([v for x in arm for v in x['old'].values()]):5.1f}%  new {100*np.mean([v for x in arm for v in x['new'].values()]):5.1f}%"
            ref = R["anchored ours_lib"]
            if not any(x is None for x in ref):
                for g in ("old", "new"):
                    x_ = {k_: v for a in arm for k_, v in a[g].items()}; y_ = {k_: v for a in ref for k_, v in a[g].items()}; ks = [q for q in x_ if q in y_]
                    d = np.array([100 * (x_[q] - y_[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                    line += f" | {g} vs ours_lib {d.mean():+.1f}pp ({int((d>0).sum())}/{int((d<0).sum())}, p={p_:.3f})"
            L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp124_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E124_STAGE", "run"); U._clear_orphaned_builds()
    {"base": base, "run": run, "report": report}[st]()
    print(f"[e124] {st} DONE", flush=True)
