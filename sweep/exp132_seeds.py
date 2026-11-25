#!/usr/bin/env python3
"""EXP-132 — training-seed robustness of the headline (10-20; paper requirement "2-3 training seeds"; keep GPUs busy).

EXP-124/125 used ONE training seed per write-protected adapter. Retrain the 22 new skills of orders 0-3 with
`train_lora.py --init s16wp --seed 1` (seed changes data order and flow-matching noise/time draws; S16 init deterministic),
same steps / batch. Then rerun the headline arms on the seed-1 adapters.

PRE-REGISTERED (10-20, before any training): adapters 22 x seed 1 (results/analysis/new_loras_s16wp_seed1); single-task 2 x 10
eps; continual sequences 0-3, t = 8, same evaluated tasks, 10 eps, states 0..9; arms ours w = 0.1, 0.25, OPCM (alpha 0.5),
averaging; 16 models. Readout: each arm seed 1 vs seed 0 (EXP-124/125 records), paired over the same task positions; pooled
over both seeds. Prediction: seed-to-seed differences small (|delta| <= 5pp on each axis for each arm); ours w=0.25 stays
>= +25pp above averaging on new skills.
Env: E132_STAGE base|run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp124_wpcontinual as W                                         # seed-0 write-protected adapters (sequences 0-3)
import exp120_privileged as P
import exp118_anchored as A
import exp116_continual as C
import exp113_controls as X
import exp112_bench as E
import unified_pool_driver as U
T = 8
ADIR = REPO / "results/analysis/new_loras_s16wp_seed1"
S1X, S1I = [], {}
for i in sorted({j for s in W.SEQS for j in A.sequence(s)[1][:8]}):
    e = E.LIB[i]; name = str(e["key"]).split(":", 1)[1]; path = ADIR / name / "adapter_model.safetensors"
    U.EXTPATH["np16wp1:" + name] = str(path)
    S1I[i] = len(E.LIB) + len(S1X); S1X.append(dict(e, key="np16wp1:" + name, tid="v" + e["tid"][1:], path=str(path)))
E.LIB.extend(S1X)
ARMS = [("ours_w0.1", "priv", 0.1), ("ours_w0.25", "priv", 0.25), ("opcm", "opcm", None), ("avg", "avg", None)]
SEED0 = {"ours_w0.1": "c124s{s}t8-wp_w0.1", "ours_w0.25": "c124s{s}t8-wp_w0.25", "opcm": "c125s{s}t8-opcm", "avg": "c125s{s}t8-avg"}

def sequence(s):
    old, new = A.sequence(s); return old, [S1I[i] for i in new]

def single():
    s = W.single()
    for i, j in S1I.items():
        rs = [E.rec(f"e132_base_{E.LIB[j]['tid']}_r{r}", E.LIB[j]["scene"]) for r in (1, 2)]; rs = [x for x in rs if x]
        if rs: s[j] = np.mean([x["success_rate"] for x in rs])
    return s

def items():
    out = []
    for s in W.SEQS:
        old, new = sequence(s); mem = old + new[:T]; ev = sorted(old[:C.NEVAL] + new[:T])
        for nm, m, w in ARMS:
            p = {"w": w, "P": new[:T]} if m == "priv" else ({"alpha": 0.5} if m == "opcm" else {})
            out.append((f"c132s{s}t{T}-{nm}", s, m, p, None, mem, ev))
    return out

def _rounds(todo, build, tagp):
    if not todo: return
    R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    cur = build(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"{tagp}_{r}"))], cwd=REPO)
        nxt = build(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e132] {tagp} round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def base():
    missing = [e["path"] for e in S1X if not pathlib.Path(e["path"]).exists()]
    if missing: raise SystemExit(f"[e132] {len(missing)} adapters not trained: {missing[:2]}")
    todo = [e for e in S1X if any(E.rec(f"e132_base_{e['tid']}_r{r}", e["scene"]) is None for r in (1, 2))]
    print(f"[e132] base: {len(todo)} adapters", flush=True)
    B = P.PBuilder()
    def build(rd):
        js = []
        for e in rd:
            out = E.PRE / f"e132_base_{e['tid']}"; B.build("ta", {"lam": 1.0}, None, [E.LIB.index(e)], out)
            js += [{"name": f"e132_base_{e['tid']}_r{r}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [],
                    "prebuilt_dir": str(out), "eval_scenes": [e["scene"]], "suite": e["suite"], "norm_stats": U.FLEET_NORM}
                   for r in (1, 2) if E.rec(f"e132_base_{e['tid']}_r{r}", e["scene"]) is None]
        return js
    _rounds(todo, build, "e132base")

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[5]), "test", it[1], it[6]) for sc in scs)]
    print(f"[e132] {len(todo)} merged models", flush=True)
    B = P.PBuilder()
    def build(rd):
        js = []
        for tg, s, m, p, c, mem, ev in rd:
            out = E.PRE / f"e132_{tg}"; t1 = time.time(); B.build(m, p, c, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e132] built {tg} in {time.time()-t1:.0f}s", flush=True)
        return js
    _rounds(todo, build, "e132")

def report():
    from scipy import stats
    sg = single()
    L = ["=== EXP-132 training-seed robustness (seed 1 vs seed 0 write-protected adapters), sequences 0-3, t = 8 ===",
         "single-task: seed 1 %.1f%% vs seed 0 %.1f%% (same 22 tasks)" % (100 * np.mean([sg[j] for j in S1I.values() if j in sg]),
                                                                        100 * np.mean([sg[W.WPI[i]] for i in S1I if W.WPI[i] in sg]))]
    for nm, m, w in ARMS:
        g1 = [W.groups(f"c132s{s}t{T}-{nm}", s, T, sequence, sg) for s in W.SEQS]
        g0 = [W.groups(SEED0[nm].format(s=s), s, T, W.sequence, sg) for s in W.SEQS]
        if any(x is None for x in g1 + g0): L.append(f"{nm}: incomplete"); continue
        line = f"{nm:11s}"
        for k_ in ("old", "new"):
            a = {q: v for z in g1 for q, v in z[k_].items()}; b = {q: v for z in g0 for q, v in z[k_].items()}; ks = [q for q in a if q in b]
            d = np.array([100 * (a[q] - b[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
            line += (f" | {k_}: seed1 {100*np.mean([a[q] for q in ks]):5.1f}% seed0 {100*np.mean([b[q] for q in ks]):5.1f}% "
                     f"(diff {d.mean():+5.1f}pp, p={p_:.3f}) pooled {100*np.mean([a[q] for q in ks] + [b[q] for q in ks]):5.1f}%")
        L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp132_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E132_STAGE", "run"); U._clear_orphaned_builds()
    {"base": base, "run": run, "report": report}[st]()
    print(f"[e132] {st} DONE", flush=True)
