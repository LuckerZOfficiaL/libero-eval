#!/usr/bin/env python3
"""EXP-127 — replication of the headline continual result (EXP-124/125) on 4 NEW sequences (10-19).

EXP-124/125 (sequences 0-3, all new tasks write-protected): ours with new tasks at w=0.25 keeps LIBERO-90 at 76.9% and the
new tasks at 50.5% after 8 additions, vs OPCM 78.4 / 36.0 (new +14.5pp, p=0.025), averaging 84.2 / 19.0, CART 84.8 / 19.2.

PRE-REGISTERED (10-19, before any rollout): sequences 4-7 (seeded like EXP-116); every new task write-protected (18 adapters
exist from EXP-123/124, 4 trained here with the same recipe); t = 4, 8; same evaluated tasks, 10 eps, states 0..9;
arms ours w = 0.1 and 0.25 (old 1/k), OPCM, averaging, CART with EXP-116's fixed rules; single-task baselines 2 x 10 eps
for the 4 new adapters. Readout: sequences 4-7 alone and pooled with 0-3 (EXP-124/125 records), ours w=0.25 vs each
baseline paired at t = 8. Prediction: the EXP-124/125 pattern replicates (ours w=0.25 > OPCM on new tasks at similar
LIBERO-90; > averaging / CART by >= 25pp on new tasks).
Env: E127_STAGE base|run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
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
NEWSEQ, ALLSEQ, TS = range(4, 8), range(0, 8), (4, 8)
ADIR = REPO / "results/analysis/new_loras_s16wp"
WPX, WPI = [], {}
for i in sorted({j for s in ALLSEQ for j in A.sequence(s)[1][:8]}):
    e = E.LIB[i]; name = str(e["key"]).split(":", 1)[1]; path = ADIR / name / "adapter_model.safetensors"
    U.EXTPATH["np16wp:" + name] = str(path)
    WPI[i] = len(E.LIB) + len(WPX); WPX.append(dict(e, key="np16wp:" + name, tid="w" + e["tid"][1:], path=str(path)))
E.LIB.extend(WPX)
BASEL = [x for x in C.METHODS if x[0] in ("avg", "cart", "opcm")]
OURS = (0.1, 0.25)

def sequence(s):
    old, new = A.sequence(s); return old, [WPI[i] for i in new]

def single():
    s = X.single()
    for i, e in WPI.items():
        rs = [E.rec(f"e123_base_{E.LIB[e]['tid']}_r{r}", E.LIB[e]["scene"]) for r in (1, 2)]; rs = [x for x in rs if x]
        if rs: s[e] = np.mean([x["success_rate"] for x in rs])
    return s

def tagof(s, t, arm):
    """records: sequences 0-3 from EXP-124 (ours) / EXP-125 (baselines), 4-7 from this experiment"""
    if s < 4: return f"c124s{s}t{t}-wp_w{arm[1]:g}" if arm[0] == "ours" else f"c125s{s}t{t}-{arm[0]}"
    return f"c127s{s}t{t}-{arm[0]}" + (f"_w{arm[1]:g}" if arm[0] == "ours" else "")

def items():
    out = []
    for s in NEWSEQ:
        old, new = sequence(s)
        for t in TS:
            mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t])
            for w in OURS: out.append((tagof(s, t, ("ours", w)), s, t, "priv", {"w": w, "P": new[:t]}, None, mem, ev))
            for m, p, c in BASEL: out.append((tagof(s, t, (m,)), s, t, m, p, c, mem, ev))
    return out

def _rounds(todo, build, tagp):
    if not todo: return
    R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    stop = lambda: bool(os.environ.get("E127_STOP_FILE")) and pathlib.Path(os.environ["E127_STOP_FILE"]).exists()
    cur = build(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"{tagp}_{r}"))], cwd=REPO)
        nxt = build(rounds[r + 1]) if r + 1 < len(rounds) and not stop() else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e127] {tagp} round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt
        if cur is None:
            if r + 1 < len(rounds): print(f"[e127] {tagp} stopped at a round boundary (stop file)", flush=True)
            break

def base():
    missing = [e["path"] for e in WPX if not pathlib.Path(e["path"]).exists()]
    if missing: raise SystemExit(f"[e127] {len(missing)} adapters not trained: {missing[:2]}")
    todo = [e for e in WPX if any(E.rec(f"e123_base_{e['tid']}_r{r}", e["scene"]) is None for r in (1, 2))]
    print(f"[e127] base: {len(todo)} adapters", flush=True)
    B = P.PBuilder()
    def build(rd):
        js = []
        for e in rd:
            out = E.PRE / f"e127_base_{e['tid']}"; B.build("ta", {"lam": 1.0}, None, [E.LIB.index(e)], out)
            js += [{"name": f"e123_base_{e['tid']}_r{r}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [],
                    "prebuilt_dir": str(out), "eval_scenes": [e["scene"]], "suite": e["suite"], "norm_stats": U.FLEET_NORM}
                   for r in (1, 2) if E.rec(f"e123_base_{e['tid']}_r{r}", e["scene"]) is None]
        return js
    _rounds(todo, build, "e127base")

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[7]) for sc in scs)]
    if os.environ.get("E127_AVAILABLE_ONLY"):                           # only models whose adapters are all trained already
        todo = [it for it in todo if all(pathlib.Path(E.LIB[i]["path"]).exists() for i in it[6] if str(E.LIB[i]["key"]).startswith("np16wp:"))]
    print(f"[e127] {len(todo)} merged models", flush=True)
    B = P.PBuilder()
    def build(rd):
        js = []
        for tg, s, t, m, p, c, mem, ev in rd:
            out = E.PRE / f"e127_{tg}"; t1 = time.time(); B.build(m, p, c, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e127] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    _rounds(todo, build, "e127")

def groups(tg, s, t, sg):
    old, new = sequence(s); ev = sorted(old[:C.NEVAL] + new[:t])
    sc = E.member_success(tg, len(old) + t, "test", s, ev)
    if len(sc) < len(ev): return None
    r = {i: sc[i] / sg[i] for i in ev if sg.get(i, 0) > 0}
    return {"old": {(s, i): v for i, v in r.items() if i < X.NF}, "new": {(s, q): r[i] for q, i in enumerate(new[:t]) if i in r}}

def report():
    from scipy import stats
    sg = single()
    arms = [("ours", w) for w in OURS] + [(m,) for m, _, _ in BASEL]
    lab = lambda a: f"ours w={a[1]:g}" if a[0] == "ours" else a[0]
    L = ["=== EXP-127 headline replication: all new tasks write-protected; sequences 4-7 alone and pooled 0-7 ===",
         "single-task (write-protected, all %d): %.1f%%" % (sum(e in sg for e in WPI.values()), 100 * np.mean([sg[e] for e in WPI.values() if e in sg]))]
    for nm, seqs in (("sequences 4-7", NEWSEQ), ("pooled 0-7", ALLSEQ)):
        for t in TS:
            G = {}
            for a in arms:
                g = [groups(tagof(s, t, a), s, t, sg) for s in seqs]
                if any(x is None for x in g): L.append(f"{nm} t={t} {lab(a):12s} incomplete"); continue
                G[a] = g
                L.append(f"{nm} t={t} {lab(a):12s} old {100*np.mean([v for x in g for v in x['old'].values()]):5.1f}%  new {100*np.mean([v for x in g for v in x['new'].values()]):5.1f}%")
            ref = G.get(("ours", 0.25))
            if t == 8 and ref:
                for a, g in G.items():
                    if a[0] == "ours": continue
                    line = f"  {nm}: ours w=0.25 vs {lab(a):6s}"
                    for k_ in ("old", "new"):
                        x_ = {q: v for z in ref for q, v in z[k_].items()}; y_ = {q: v for z in g for q, v in z[k_].items()}; ks = [q for q in x_ if q in y_]
                        d = np.array([100 * (x_[q] - y_[q]) for q in ks]); p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                        line += f" | {k_} {d.mean():+6.1f}pp ({int((d>0).sum())}/{int((d<0).sum())}, p={p_:.4f}, n={len(ks)})"
                    L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp127_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E127_STAGE", "run"); U._clear_orphaned_builds()
    {"base": base, "run": run, "report": report}[st]()
    print(f"[e127] {st} DONE", flush=True)
