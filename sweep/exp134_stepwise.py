#!/usr/bin/env python3
"""EXP-134 — make the continual claim real: evaluate after EVERY addition (user plan 10-21).

So far only the final model (t = 8; t = 4 in the log) was evaluated; for our sum-based merge that equals a one-shot merge.
Here: 16 library (LIBERO-90) skills, then 8 new skills added one at a time (4 orders, EXP-116 sequences); after EVERY
addition t = 1..8 (and t = 0) the merged model is evaluated on ALL 16 library skills + all new skills so far, 10 episodes,
states 0..9.

PRE-REGISTERED (10-21, before any rollout):
  settings   wp      : new skills write-protected (EXP-124 adapters)            -> arms ours w=0.25, ours w=0.1,
                       ours budget (w_t = min(0.25, 1 / n_new): all new skills together never weigh more than one skill),
                       OPCM (alpha 0.5), RETAIN-style (a 0.5), averaging
             scratch : new skills trained from scratch (EXP-116 adapters)       -> arms ours (1/k), OPCM, RETAIN-style, averaging
             t = 0 (16 library skills only) is shared by both settings: ours (= every ours variant), averaging (= RETAIN), OPCM
  metrics    per t: library retention (16 x 4 orders), new-skill retention, overall retention (all skills in the model);
             average incremental retention = mean over t = 1..8 of overall retention; library forgetting curve = library
             retention(t) - library retention(0); backward transfer of new skill j = retention at t = 8 minus at t = j
  predictions wp: ours w=0.25 has the highest average incremental new-skill retention; ours' library curve is flat for w=0.1
             and declines roughly linearly with t for w=0.25; the budget schedule flattens the library curve vs w=0.25 at a
             cost on new skills that grows with t; RETAIN forgets early new skills (negative backward transfer), ours does not.
             scratch: ours' library curve stays flat; OPCM / RETAIN decline with t; averaging declines steadily.
Env: E134_SETTING wp|scratch, E134_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp128_retain as RT                                            # RBuilder (retain + priv + CBuilder methods); registers WP
import exp124_wpcontinual as W
import exp116_continual as C
import exp112_bench as E
import exp113_controls as X
import unified_pool_driver as U
SETTING = os.environ.get("E134_SETTING", "wp")
SEQF = W.sequence if SETTING == "wp" else C.sequence
PFX = "c134w" if SETTING == "wp" else "c134z"
TS = [int(x) for x in os.environ.get("E134_TS", "1,2,3,4,5,6,7,8").split(",")]   # 10-21 trimmed plan: scratch at 2,4,6,8
ARMS = {"wp": [("ours_w0.25", "priv", 0.25), ("opcm", "opcm", None), ("avg", "avg", None), ("retain", "retain", None),
               ("ours_w0.1", "priv", 0.1), ("ours_budget", "priv", "budget")],
        "scratch": [("ours", "ours_lib", None), ("opcm", "opcm", None), ("retain", "retain", None), ("avg", "avg", None)]}[SETTING]
ARMS = [a for a in ARMS if a[0] not in os.environ.get("E134_SKIP", "").split(",")]   # 10-21 trimmed plan: wp skips retain
T0 = [("ours", "ours_lib"), ("avg", "avg"), ("opcm", "opcm")]                # t = 0 models (no new skill), shared by settings

def params(m, w, t, old, new):
    if m == "priv": return {"w": min(0.25, 1.0 / t) if w == "budget" else w, "P": new[:t]}
    if m == "opcm": return {"alpha": 0.5}
    if m == "retain": return {"a": 0.5, "n_old": len(old)}
    return {}

def items():
    out = []
    for s in W.SEQS:                                                     # t = 0 first (shared)
        old, new = SEQF(s)
        for nm, m in T0: out.append((f"c134s{s}t0-{nm}", s, 0, m, {"alpha": 0.5} if m == "opcm" else {}, list(old), sorted(old)))
    for nm, m, w in ARMS:                                                # then arm by arm (priority order)
        for s in W.SEQS:
            old, new = SEQF(s)
            for t in TS:
                out.append((f"{PFX}s{s}t{t}-{nm}", s, t, m, params(m, w, t, old, new), old + new[:t], sorted(old + new[:t])))
    return out

def run():
    allit = items(); idx = {it[0]: q for q, it in enumerate(allit)}
    todo = [it for it in allit if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[5]), "test", it[1], it[6]) for sc in scs)]
    if os.environ.get("E134_SHARD"):                                     # "k,..,k/n": keep items whose index % n is listed
        keep, n_ = os.environ["E134_SHARD"].split("/"); keep = {int(x) for x in keep.split(",")}
        todo = [it for it in todo if idx[it[0]] % int(n_) in keep]
    print(f"[e134 {SETTING}] {len(todo)} merged models, {sum(len(x[6]) for x in todo)} evaluations", flush=True)
    if not todo: return
    B = RT.RBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, m, p, mem, ev in rd:
            out = E.PRE / f"e134_{tg}"; t1 = time.time(); B.build(m, p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e134] built {tg} ({len(mem)} skills) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e134{SETTING}{os.environ.get('E134_SHARD', '').replace('/', 'of').replace(',', '')}_{r}"))], cwd=REPO)
        try: nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        except Exception:                                                # 10-24: never orphan a running eval (it holds the ports)
            proc.wait()                                                  # nor leave its models in /dev/shm
            for d in {j["prebuilt_dir"] for j in cur} | {str(E.PRE / f"e134_{it[0]}") for it in rounds[r + 1]}: shutil.rmtree(d, ignore_errors=True)
            raise
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e134] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def ret(tg, s, t, sg):
    """per-skill retention in one model: {('old'|'new', key): value}; new keyed by arrival position j (1-based)"""
    old, new = SEQF(s); ev = sorted(old + new[:t])
    sc = E.member_success(tg, len(old) + t, "test", s, ev)
    if len(sc) < len(ev): return None
    out = {("old", (s, i)): sc[i] / sg[i] for i in old if sg.get(i, 0) > 0}
    out.update({("new", (s, j + 1)): sc[i] / sg[i] for j, i in enumerate(new[:t]) if sg.get(i, 0) > 0})
    return out

def t0tag(nm, s):
    base = {"ours_w0.25": "ours", "ours_w0.1": "ours", "ours_budget": "ours", "ours": "ours", "retain": "avg"}.get(nm, nm)
    return f"c134s{s}t0-{base}"

def report():
    sg = W.single(); res = {}
    L = [f"=== EXP-134 ({SETTING}) evaluation after every addition; 4 orders; retention ===",
         f"{'arm':12s} " + " ".join(f"{'t=' + str(t):>13s}" for t in range(0, 9)) + "   (library % / new %)"]
    for nm, m, w in ARMS:
        per_t = {}
        for t in range(0, 9):
            rr = [ret(t0tag(nm, s) if t == 0 else f"{PFX}s{s}t{t}-{nm}", s, t, sg) for s in W.SEQS]
            if any(r is None for r in rr): continue
            d = {k: v for r in rr for k, v in r.items()}
            per_t[t] = {"lib": float(np.mean([v for (g, _), v in d.items() if g == "old"])),
                        "new": float(np.mean([v for (g, _), v in d.items() if g == "new"])) if t else None,
                        "all": float(np.mean(list(d.values()))), "raw": {f"{g}|{k}": v for (g, k), v in d.items()}}
        res[nm] = per_t
        cells = []
        for t in range(0, 9):
            if t not in per_t: cells.append(f"{'—':>13s}"); continue
            n = per_t[t]["new"]; cells.append(f"{100*per_t[t]['lib']:5.1f} / {'  -  ' if n is None else f'{100*n:5.1f}'}")
        L.append(f"{nm:12s} " + " ".join(f"{c:>13s}" for c in cells))
    L.append("summary: average incremental retention (mean over t=1..8, overall / new), library forgetting at t=8, mean new-skill backward transfer")
    for nm, per_t in res.items():
        if not all(t in per_t for t in [0] + TS): L.append(f"  {nm:12s} incomplete"); continue
        aia = np.mean([per_t[t]["all"] for t in TS]); ain = np.mean([per_t[t]["new"] for t in TS])   # over the evaluated steps
        fg = per_t[8]["lib"] - per_t[0]["lib"]
        bwt = [per_t[8]["raw"][f"new|({s}, {j})"] - per_t[j]["raw"][f"new|({s}, {j})"] for s in W.SEQS for j in TS if j < 8
               if f"new|({s}, {j})" in per_t[8]["raw"] and f"new|({s}, {j})" in per_t[j]["raw"]]
        L.append(f"  {nm:12s} avg incremental {100*aia:5.1f}% overall / {100*ain:5.1f}% new | library forgetting {100*fg:+5.1f}pp | new-skill BWT {100*np.mean(bwt):+5.1f}pp (n={len(bwt)})")
    txt = "\n".join(L); print(txt)
    (REPO / f"results/analysis/exp134_{SETTING}_report.txt").write_text(txt + "\n")
    json.dump({k: {str(t): {kk: vv for kk, vv in v.items() if kk != "raw"} for t, v in pt.items()} for k, pt in res.items()},
              open(REPO / f"results/analysis/exp134_{SETTING}_curves.json", "w"), indent=1)

if __name__ == "__main__":
    st = os.environ.get("E134_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e134] {st} DONE", flush=True)
