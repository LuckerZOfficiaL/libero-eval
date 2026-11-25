#!/usr/bin/env python3
"""EXP-129 — out-of-distribution robustness: merged models vs single adapters ("why merge rather than route?"), 10-19.

MergeVLA (CVPR'26) routes each episode to a task-specific mask/head; in a language-conditioned VLA one could route by the
instruction to the task's own adapter and recover single-task success (the oracle-routing upper bound). RETAIN (ICLR'26)
reports that weight merging buys OOD robustness. Question: under object-position shifts outside the demonstrations, how
much of the single-adapter (routing) advantage survives, and does merging lose less from ID to OOD?

PRE-REGISTERED (10-19, before any rollout):
  perturbation `ood_shift` (EXP-81, validated): every free-joint object's xy translated by d metres along one
             episode-seeded random direction, layout preserved; canonical states span ~+-2.4 cm. d in {0.05, 0.10};
             d = 0 = existing ID records. States 0..9, 10 episodes per evaluation.
  merged     EXP-124 setting (sequences 0-3, every new task write-protected, t = 8): ours w=0.25, ours w=0.1, plain
             averaging; evaluated tasks = 8 LIBERO-90 + 8 new per sequence (same as EXP-124/125)
  single     the oracle-routing bound: each evaluated task's own adapter alone (fleet adapter for LIBERO-90 tasks, its
             write-protected adapter for new tasks), 2 x 10 episodes, same shifts
  readout    success of merged vs single per task instance (paired), per shift; "OOD drop" = ID - OOD success, paired
             drop(single) - drop(merged); retention merged / single per shift
  predictions single adapters stay above the merged models at every shift (routing bound), but the gap SHRINKS under shift
             (merged models drop less, RETAIN-like), most for LIBERO-90 tasks inside the merged library
Env: E129_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp124_wpcontinual as W
import exp120_privileged as P
import exp116_continual as C
import exp113_controls as X
import exp112_bench as E
import unified_pool_driver as U
SHIFTS, T = (0.05, 0.10), 8
ARMS = [("ours_w0.25", {"w": 0.25}), ("ours_w0.1", {"w": 0.1}), ("avg", None)]

def stag(d): return f"d{int(round(d * 100)):02d}"

def merged_items(d):
    out = []
    for s in W.SEQS:
        old, new = W.sequence(s); mem = old + new[:T]; ev = sorted(old[:C.NEVAL] + new[:T])
        for nm, p in ARMS:
            m, pp = ("priv", {"w": p["w"], "P": new[:T]}) if p else ("avg", {})
            out.append(("merged", f"c129{stag(d)}s{s}t{T}-{nm}", s, m, pp, mem, ev))
    return out

def single_tasks():
    ts = set()
    for s in W.SEQS:
        old, new = W.sequence(s); ts |= set(old[:C.NEVAL]) | set(new[:T])
    return sorted(ts)

def single_items(d):
    return [("single", f"e129{stag(d)}_single_{E.LIB[i]['tid']}", None, "ta", {"lam": 1.0}, [i], [i]) for i in single_tasks()]

def jobs_of(it):
    kind, tg, s, m, p, mem, ev = it
    if kind == "single":
        i = mem[0]; return [(f"{tg}_r{r}", E.LIB[i]["suite"], [E.LIB[i]["scene"]]) for r in (1, 2)]
    return [(nm, su, scs) for nm, su, scs, ch in E.member_jobs(tg, len(mem), "test", s, ev)]

def run():
    B = P.PBuilder()
    for d in SHIFTS:
        todo = [it for it in merged_items(d) + single_items(d) if any(E.rec(nm, sc) is None for nm, su, scs in jobs_of(it) for sc in scs)]
        print(f"[e129] shift {d}: {len(todo)} models", flush=True)
        if not todo: continue
        R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
        def build(rd):
            js = []
            for it in rd:
                kind, tg, s, m, p, mem, ev = it
                out = E.PRE / f"e129_{tg}"; t1 = time.time(); B.build(m, p, None, mem, out)
                for nm, su, scs in jobs_of(it):
                    if all(E.rec(nm, sc) is not None for sc in scs): continue
                    js.append({"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                               "eval_scenes": scs, "suite": su, "norm_stats": U.FLEET_NORM})
                print(f"[e129] built {tg} in {time.time()-t1:.0f}s", flush=True)
            return js
        def plan(js, r):
            pth = E.plan(js, 10, 0, f"e129{stag(d)}_{r}"); meta = json.load(open(pth)); meta["ood_shift"] = d
            json.dump(meta, open(pth, "w"), indent=1); return pth
        cur = build(rounds[0])
        for r in range(len(rounds)):
            proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(plan(cur, r))], cwd=REPO)
            nxt = build(rounds[r + 1]) if r + 1 < len(rounds) else None
            proc.wait()
            for dd in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(dd, ignore_errors=True)
            print(f"[e129] shift {d} round {r + 1}/{len(rounds)} done", flush=True)
            cur = nxt

def succ(d, sg):
    """per (sequence, task) instance: {'merged': {arm: v}, 'single': v}; d = 0 uses the existing ID records"""
    out = {}
    for s in W.SEQS:
        old, new = W.sequence(s); ev = sorted(old[:C.NEVAL] + new[:T]); k = len(old) + T
        for nm, p in ARMS:
            tg = (f"c124s{s}t{T}-wp_w{p['w']:g}" if p else f"c125s{s}t{T}-avg") if d == 0 else f"c129{stag(d)}s{s}t{T}-{nm}"
            sc = E.member_success(tg, k, "test", s, ev)
            for i in ev:
                if i in sc: out.setdefault((s, i), {"merged": {}})["merged"][nm] = sc[i]
        for i in ev:
            if d == 0: v = sg.get(i)
            else:
                rs = [E.rec(f"e129{stag(d)}_single_{E.LIB[i]['tid']}_r{r}", E.LIB[i]["scene"]) for r in (1, 2)]; rs = [x["success_rate"] for x in rs if x]
                v = float(np.mean(rs)) if rs else None
            if (s, i) in out and v is not None: out[(s, i)]["single"] = v
    return out

def report():
    from scipy import stats
    sg = W.single()
    S = {d: succ(d, sg) for d in (0.0,) + SHIFTS}
    L = ["=== EXP-129 OOD (object shift) — merged models vs single adapters (oracle routing), EXP-124 setting, t = 8 ==="]
    for grp in ("LIBERO-90", "new"):
        L.append(f"-- {grp} tasks --")
        for d, D in S.items():
            ks = [q for q, v in D.items() if "single" in v and len(v["merged"]) == len(ARMS) and ((q[1] < X.NF) == (grp == "LIBERO-90"))]
            if not ks: L.append(f"  shift {d:.2f}: incomplete"); continue
            line = f"  shift {d:.2f} (n={len(ks)}): single {100*np.mean([D[q]['single'] for q in ks]):5.1f}%"
            for nm, _ in ARMS:
                dd = np.array([100 * (D[q]["merged"][nm] - D[q]["single"]) for q in ks]); p_ = stats.wilcoxon(dd).pvalue if np.any(dd) else float("nan")
                line += f" | {nm} {100*np.mean([D[q]['merged'][nm] for q in ks]):5.1f}% ({dd.mean():+5.1f} vs single, p={p_:.3f})"
            L.append(line)
        for d in SHIFTS:                                                # paired OOD drop: single vs merged
            ks = [q for q in S[d] if q in S[0.0] and "single" in S[d][q] and "single" in S[0.0][q] and len(S[d][q]["merged"]) == len(ARMS)
                  and len(S[0.0][q]["merged"]) == len(ARMS) and ((q[1] < X.NF) == (grp == "LIBERO-90"))]
            if not ks: continue
            line = f"  drop ID->{d:.2f}: single {100*np.mean([S[0.0][q]['single'] - S[d][q]['single'] for q in ks]):+5.1f}pp"
            for nm, _ in ARMS:
                x_ = np.array([100 * ((S[0.0][q]["single"] - S[d][q]["single"]) - (S[0.0][q]["merged"][nm] - S[d][q]["merged"][nm])) for q in ks])
                p_ = stats.wilcoxon(x_).pvalue if np.any(x_) else float("nan")
                line += f" | {nm} {100*np.mean([S[0.0][q]['merged'][nm] - S[d][q]['merged'][nm] for q in ks]):+5.1f}pp (single minus merged drop {x_.mean():+5.1f}, p={p_:.3f})"
            L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp129_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E129_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e129] {st} DONE", flush=True)
