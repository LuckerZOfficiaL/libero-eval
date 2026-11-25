#!/usr/bin/env python3
"""EXP-113 — mechanism controls for the EXP-112 advantage of ours over averaging (k = 8).

PRE-REGISTERED (10-13, before any rollout):
  arms     avg; ours = pool S16 + alpha * sum r_i with alpha = 1/8 (EXP-112's validated choice at k=8; no new tuning)
  library  EXP-112's 104 tasks + the 30 S16-INIT versions of the external tasks (same tasks, trained from the shared part)
  conditions (12 seeded subsets each; 10 episodes per member on states 0..9):
    A  8 fleet, 0 external                 B  4 fleet + 4 zero-init external        C  8 zero-init external
    D  EXP-112's k=8 test subsets with their 2 external members replaced by the S16-init version of the same task
  predictions: A |ours-avg| small; B > +7.6pp (EXP-112, 2 of 8); C ours-avg >= -3pp; D shrinks toward 0 vs EXP-112
  single-task baselines for the 30 S16-init externals: 2 x 10 episodes on states 0..9
Records e113_*. Env: E113_STAGE base|run|report, E113_CONDS (A,B,C,D), E113_GPUS, E113_PER_GPU, E113_DEV, E113_ROUND.
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("E112_DEV", os.environ.get("E113_DEV", "cuda:1"))
os.environ.setdefault("E112_GPUS", os.environ.get("E113_GPUS", "0,1,2"))
if os.environ.get("E113_PER_GPU"): os.environ["E112_PER_GPU"] = os.environ["E113_PER_GPU"]
import exp112_bench as E
import unified_pool_driver as U
K = 8; ARMS = [("avg", {}, None), ("ours", {"alpha": 1 / 8}, None)]
NF, NE = len(E.FLEET), len(E.EXT)
ORIG_TEST = {k: E.subsets(k, "test") for k in (2, 4, 8)}; ORIG_VAL = {k: E.subsets(k, "val") for k in (2, 4, 8)}
ORIG_TEST8 = ORIG_TEST[8]
# 10-14 fix: E.subsets() must keep returning EXP-112's subsets after the library grows (best_settings / validation reads)
E.subsets = lambda k, split: (ORIG_TEST if split == "test" else ORIG_VAL)[k]                                   # EXP-112's k=8 test subsets, BEFORE the library
                                                                     # is extended (E.subsets samples externals from
                                                                     # everything past the fleet -- 10-13 bug fix)
S16X = []
for e in E.EXT:                                                     # S16-init twin of every zero-init external task
    name = e["key"].split(":", 1)[1]; path = REPO / "results/analysis/new_loras_s16" / name / "adapter_model.safetensors"
    U.EXTPATH["np16:" + name] = str(path)
    S16X.append(dict(e, key="np16:" + name, tid="s" + e["tid"][1:], path=str(path)))
E.LIB.extend(S16X)                                                  # indices NF+NE .. NF+2NE-1 (same task order as EXT)

def subsets(cond):
    rng = np.random.default_rng(11300 + ord(cond))
    fi, ei = list(range(NF)), list(range(NF, NF + NE))
    if cond == "A": return [sorted(int(x) for x in rng.choice(fi, 8, replace=False)) for _ in range(12)]
    if cond == "B": return [sorted([int(x) for x in rng.choice(fi, 4, replace=False)] + [int(x) for x in rng.choice(ei, 4, replace=False)]) for _ in range(12)]
    if cond == "C": return [sorted(int(x) for x in rng.choice(ei, 8, replace=False)) for _ in range(12)]
    if cond == "D": return [sorted(i + NE if i >= NF else i for i in S) for S in ORIG_TEST8]
    raise ValueError(cond)

def tg(cond, m, p, c): return f"c113{cond}-{E.tag(m, p, c)}"

def run(conds):
    items = [(cond, m, p, c, j, S) for cond in conds for j, S in enumerate(subsets(cond)) for m, p, c in ARMS
             if any(E.rec(nm, s) is None for nm, su, scs, ch in E.member_jobs(tg(cond, m, p, c), K, "test", j, S) for s in scs)]
    print(f"[e113] {len(items)} merged models for conditions {conds}", flush=True)
    B = E.Builder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [items[r:r + R] for r in range(0, len(items), R)]
    def build_round(rd):
        js = []
        for cond, m, p, c, j, S in rd:
            t = tg(cond, m, p, c); out = E.PRE / f"e113_{t}_{j}"; t1 = time.time(); B.build(m, p, c, S, out)
            for name, su, scenes, ch in E.member_jobs(t, K, "test", j, S):
                if all(E.rec(name, s) is not None for s in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e113] built {t} subset {j} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0]) if rounds else []
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e113_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e113] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def single():
    s = {}
    for i, e in enumerate(E.LIB):
        pre = "e113" if str(e["key"]).startswith("np16:") else "e112"
        rs = [E.rec(f"{pre}_base_{e['tid']}_r{r}", e["scene"]) for r in (1, 2)]; rs = [x for x in rs if x]
        if rs: s[i] = np.mean([x["success_rate"] for x in rs])
    return s

def report(conds):
    from scipy import stats
    sg = single(); L = ["=== EXP-113 mechanism controls: ours (alpha=1/8) vs averaging, k=8, 12 subsets per condition, 10 eps per member ==="]
    def ret(cond, m, p, c, j, S, which):
        s = E.member_success(tg(cond, m, p, c), K, "test", j, S)
        v = [s[i] / sg[i] for i in S if i in s and sg.get(i, 0) > 0 and (which == "all" or (which == "fleet") == (i < NF))]
        return np.mean(v) if v else np.nan
    for cond in conds:
        rows = []
        for j, S in enumerate(subsets(cond)):
            rows.append([ret(cond, m, p, c, j, S, w) for w in ("all", "fleet", "ext") for m, p, c in ARMS])
        a = np.array(rows, float)                                              # columns: all avg, all ours, fleet avg, fleet ours, ext avg, ext ours
        d = 100 * (a[:, 1] - a[:, 0]); ok = np.isfinite(d)
        p_ = stats.wilcoxon(d[ok]).pvalue if ok.sum() > 1 and np.any(d[ok]) else np.nan
        L.append(f"{cond}: avg {100*np.nanmean(a[:,0]):5.1f}%  ours {100*np.nanmean(a[:,1]):5.1f}%  ours-avg {np.nanmean(d):+5.1f}pp "
                 f"(wins {int((d[ok]>0).sum())}/{ok.sum()}, Wilcoxon p={p_:.3f}) | fleet members {100*np.nanmean(a[:,3]-a[:,2]):+5.1f}pp, "
                 f"external members {100*np.nanmean(a[:,5]-a[:,4]):+5.1f}pp")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp113_report.txt").write_text(txt + "\n")

def main():
    stage = os.environ.get("E113_STAGE", "run"); conds = os.environ.get("E113_CONDS", "A,B,C,D").split(","); U._clear_orphaned_builds()
    if stage == "base":
        J = [{"name": f"e113_base_{e['tid']}_r{r}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [e["path"]],
              "eval_scenes": [e["scene"]], "suite": e["suite"], "norm_stats": U.FLEET_NORM}
             for r in (1, 2) for e in S16X if E.rec(f"e113_base_{e['tid']}_r{r}", e["scene"]) is None]
        print(f"[e113] base: {len(J)} single-task jobs (S16-init externals)", flush=True)
        if J: subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(J, 10, 0, "e113base"))], cwd=REPO)
    elif stage == "run": run(conds)
    else: report(conds)
    print(f"[e113] {stage} DONE", flush=True)

if __name__ == "__main__":
    main()
