#!/usr/bin/env python3
"""EXP-131 — matched-retention comparison vs OPCM (10-19; user: compare at matched old-skill cost; keep GPUs busy).

Pooled over 8 sequences (EXP-124/125/127), ours with all new tasks write-protected runs from (78.0 / 31.4) at w = 0.1 to
(71.9 / 50.7) at w = 0.25 (LIBERO-90 / new retention after 8 additions); OPCM sits at (74.8 / 41.2) — linear interpolation
of our two points passes roughly THROUGH OPCM's point. Question: at matched LIBERO-90 retention, does our curve dominate
OPCM's? Both methods get a knob: ours w; OPCM its energy threshold alpha (official default 0.5).

PRE-REGISTERED (10-19, before any rollout): sequences 0-7, every new task write-protected, t = 8, same evaluated tasks,
10 eps, states 0..9. New arms: ours w = 0.15, 0.2; OPCM alpha = 0.3, 0.7 (official norm rescaling). Existing: ours w = 0.1,
0.25 and OPCM alpha 0.5 (EXP-124/125/127). Readout: both curves (LIBERO-90 vs new retention, pooled over 8 sequences);
new-task retention of each method interpolated at the other's LIBERO-90 levels; paired tests at the nearest matched points
(ours w vs OPCM alpha with the closest LIBERO-90). Prediction: our curve lies on or slightly above OPCM's (no clear
dominance); OPCM's alpha moves it little.
Env: E131_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp127_replicate_wp as R                                       # registers all 28 write-protected adapters
import exp120_privileged as P
import exp116_continual as C
import exp112_bench as E
import unified_pool_driver as U
T, SEQS = 8, R.ALLSEQ
NEW = [("ours", 0.15), ("ours", 0.2), ("opcm", 0.3), ("opcm", 0.7)]

def tagof(s, arm):
    m, v = arm
    if (m, v) in (("ours", 0.1), ("ours", 0.25)) or (m, v) == ("opcm", 0.5):
        return R.tagof(s, T, ("ours", v) if m == "ours" else ("opcm",))
    return f"c131s{s}t{T}-{m}_{'w' if m == 'ours' else 'a'}{v:g}"

def items():
    out = []
    for s in SEQS:
        old, new = R.sequence(s); mem = old + new[:T]; ev = sorted(old[:C.NEVAL] + new[:T])
        for m, v in NEW:
            if m == "ours": out.append((tagof(s, (m, v)), s, "priv", {"w": v, "P": new[:T]}, None, mem, ev))
            else: out.append((tagof(s, (m, v)), s, "opcm", {"alpha": v}, None, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[5]), "test", it[1], it[6]) for sc in scs)]
    print(f"[e131] {len(todo)} merged models", flush=True)
    if not todo: return
    B = P.PBuilder(); Rn = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + Rn] for r in range(0, len(todo), Rn)]
    def build_round(rd):
        js = []
        for tg, s, m, p, c, mem, ev in rd:
            out = E.PRE / f"e131_{tg}"; t1 = time.time(); B.build(m, p, c, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e131] built {tg} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e131_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e131] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    sg = R.single()
    curves = {"ours": [("ours", w) for w in (0.1, 0.15, 0.2, 0.25)], "OPCM": [("opcm", a) for a in (0.3, 0.5, 0.7)]}
    L = ["=== EXP-131 matched-retention comparison, pooled sequences 0-7, t = 8 (LIBERO-90 / new retention) ==="]
    G, pts = {}, {}
    for cname, arms in curves.items():
        for a in arms:
            g = [R.groups(tagof(s, a), s, T, sg) for s in SEQS]
            if any(x is None for x in g): L.append(f"{a}: incomplete"); continue
            G[a] = g; o = 100 * np.mean([v for x in g for v in x["old"].values()]); n = 100 * np.mean([v for x in g for v in x["new"].values()])
            pts.setdefault(cname, []).append((o, n, a)); L.append(f"  {cname:4s} {a[0]} {a[1]:<5g} LIBERO-90 {o:5.1f}%  new {n:5.1f}%")
    if "ours" in pts and "OPCM" in pts:
        po = sorted(pts["ours"])                                       # by LIBERO-90
        for o, n, a in pts["OPCM"]:
            if po[0][0] <= o <= po[-1][0]:
                n_int = float(np.interp(o, [p[0] for p in po], [p[1] for p in po]))
                L.append(f"  at OPCM alpha={a[1]:g}'s LIBERO-90 {o:.1f}%: ours (interpolated) new {n_int:.1f}% vs OPCM {n:.1f}% ({n_int - n:+.1f}pp)")
            near = min(pts["ours"], key=lambda q: abs(q[0] - o))
            x_ = {q: v for z in G[near[2]] for q, v in z["new"].items()}; y_ = {q: v for z in G[a] for q, v in z["new"].items()}; ks = [q for q in x_ if q in y_]
            xo = {q: v for z in G[near[2]] for q, v in z["old"].items()}; yo = {q: v for z in G[a] for q, v in z["old"].items()}; ko = [q for q in xo if q in yo]
            d = np.array([100 * (x_[q] - y_[q]) for q in ks]); do = np.array([100 * (xo[q] - yo[q]) for q in ko])
            L.append(f"  nearest pair: ours w={near[2][1]:g} vs OPCM alpha={a[1]:g}: new {d.mean():+.1f}pp (p={stats.wilcoxon(d).pvalue if np.any(d) else float('nan'):.4f}), "
                     f"LIBERO-90 {do.mean():+.1f}pp (p={stats.wilcoxon(do).pvalue if np.any(do) else float('nan'):.4f})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp131_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E131_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e131] {st} DONE", flush=True)
