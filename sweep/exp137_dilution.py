#!/usr/bin/env python3
"""EXP-137 — why does the library decline under our training-free merge: smaller weight or interference? (user 10-24)

EXP-134 (scratch): ours' library retention 86.3 -> 76.6 over 8 additions. Our merge W = W0 + S + sum_i (1/k) r_i weights every
skill's task part 1/k, so each library skill's part shrinks 1/16 -> 1/24 as skills arrive, even with zero interference.
Separate the two causes (same shared part S, same orders / adapters / evaluation as EXP-134 scratch):
  fixlib  library parts fixed at 1/16, new parts at 1/k exactly as in ours   -> only the library's shrinkage removed
  dilute  library parts at 1/(16 + t), NO new skill in the model             -> only the shrinkage, no interference
  (ours = EXP-134 c134zs{s}t{t}-ours; t = 0 = c134s{s}t0-ours, where all three coincide)
PRE-REGISTERED (10-24, before any rollout): fixlib at t = 2, 4, 6, 8 (16 library + new so far), dilute at t = 4, 8 (16 library
skills); 4 orders, 10 eps, states 0..9; 24 models, 464 evaluations. Readout: library retention per t for ours / fixlib / dilute;
fixlib - ours (paired over order x library skill) = effect of the shrinkage in the presence of new skills; dilute - t0 = the
shrinkage alone; fixlib - t0 = interference of the new skills at a fixed library weight; new-skill retention fixlib vs ours.
Predictions: (1) fixlib's library curve is near flat (t = 8 within 4pp of t = 0); (2) dilute at t = 8 accounts for most of
ours' drop (<= -6pp of the -9.8pp); (3) new-skill retention fixlib within 3pp of ours (same new-skill weights).
Env: E137_STAGE run|report; E137_SHARD k/n; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp120_privileged as P
import exp124_wpcontinual as W
import exp116_continual as C
import exp113_controls as X
import exp112_bench as E
import unified_pool_driver as U
ARMS = [("fixlib", (2, 4, 6, 8)), ("dilute", (4, 8))]

def items():
    out = []
    for t in (2, 4, 6, 8):                                               # interleaved -> shards balance
        for nm, ts in ARMS:
            if t not in ts: continue
            for s in W.SEQS:
                old, new = C.sequence(s)
                if nm == "fixlib": p, mem = {"w": 1.0 / len(old), "P": list(old)}, old + new[:t]
                else: p, mem = {"w": 1.0 / (len(old) + t), "P": list(old)}, list(old)
                out.append((f"c137s{s}t{t}-{nm}", s, t, p, mem, sorted(mem)))
    return out

def run():
    allit = items()
    todo = [it for it in allit if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[4]), "test", it[1], it[5]) for sc in scs)]
    if os.environ.get("E137_SHARD"):
        k_, n_ = (int(x) for x in os.environ["E137_SHARD"].split("/")); todo = [it for it in todo if allit.index(it) % n_ == k_]
    print(f"[e137] {len(todo)} merged models, {sum(len(x[5]) for x in todo)} evaluations", flush=True)
    if not todo: return
    B = P.PBuilder(); R = int(os.environ.get("E113_ROUND", "1")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, p, mem, ev in rd:
            out = E.PRE / f"e137_{tg}"; t1 = time.time(); B.build("priv", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e137] built {tg} ({len(mem)} skills) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e137{os.environ.get('E137_SHARD', '').replace('/', 'of')}_{r}"))], cwd=REPO)
        try: nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        except Exception:                                                # never orphan a running eval nor leave its models
            proc.wait()
            for d in {j["prebuilt_dir"] for j in cur} | {str(E.PRE / f"e137_{it[0]}") for it in rounds[r + 1]}: shutil.rmtree(d, ignore_errors=True)
            raise
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e137] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def ret(tg, s, t, k, sg):
    """per-skill retention {('old'|'new', (s, i)): value} in one model with k members, evaluated on library + new[:t]"""
    old, new = C.sequence(s); ev = sorted(old + new[:t])
    sc = E.member_success(tg, k, "test", s, ev)
    if len(sc) < len(ev): return None
    out = {("old", (s, i)): sc[i] / sg[i] for i in old if sg.get(i, 0) > 0}
    out.update({("new", (s, i)): sc[i] / sg[i] for i in new[:t] if sg.get(i, 0) > 0})
    return out

def report():
    from scipy import stats
    sg = W.single(); res = {}
    tags = {"ours": lambda s, t: (f"c134s{s}t0-ours" if t == 0 else f"c134zs{s}t{t}-ours", 16 + t, t),
            "fixlib": lambda s, t: (f"c137s{s}t{t}-fixlib", 16 + t, t), "dilute": lambda s, t: (f"c137s{s}t{t}-dilute", 16, 0)}
    for nm, f in tags.items():
        for t in range(0, 9):
            rr = []
            for s in W.SEQS:
                tg, k, te = f(s, t) if t else (f"c134s{s}t0-ours", 16, 0)
                rr.append(ret(tg, s, te, k, sg))
            if all(r is not None for r in rr): res[(nm, t)] = {k: v for r in rr for k, v in r.items()}
    m = lambda d, g: 100 * np.mean([v for (g_, _), v in d.items() if g_ == g])
    L = ["=== EXP-137 library decline: smaller weight (dilution) vs interference; scratch adapters, 4 orders; retention ===",
         "arm      " + " ".join(f"{'t=' + str(t):>13s}" for t in range(0, 9)) + "   (library % / new %)"]
    for nm in tags:
        cells = [(f"{m(res[(nm, t)], 'old'):5.1f} / " + (f"{m(res[(nm, t)], 'new'):5.1f}" if nm != "dilute" and t else "  -  ")) if (nm, t) in res else "—" for t in range(0, 9)]
        L.append(f"{nm:8s} " + " ".join(f"{c:>13s}" for c in cells))
    L.append("paired over (order, library skill), pp:")
    for t in (2, 4, 6, 8):
        for a, b in [("fixlib", "ours"), ("fixlib", 0), ("dilute", 0), ("ours", 0)]:
            ka, kb = (a, t), ((b, t) if b else ("ours", 0))
            if ka not in res or kb not in res: continue
            ks = [k for k in res[ka] if k[0] == "old" and k in res[kb]]; d = np.array([100 * (res[ka][k] - res[kb][k]) for k in ks])
            p_ = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
            L.append(f"  t={t} {a} - {'t=0' if not b else b}: {d.mean():+5.1f} (n={len(d)}, p={p_:.2g})")
        if ("fixlib", t) in res and ("ours", t) in res:
            ks = [k for k in res[("fixlib", t)] if k[0] == "new" and k in res[("ours", t)]]
            d = np.array([100 * (res[("fixlib", t)][k] - res[("ours", t)][k]) for k in ks])
            L.append(f"  t={t} new skills fixlib - ours: {d.mean():+5.1f} (n={len(d)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp137_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E137_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e137] {st} DONE", flush=True)
