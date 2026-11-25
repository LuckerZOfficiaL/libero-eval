#!/usr/bin/env python3
"""EXP-57 analysis — per-state difficulty, and an HONEST guard built on state-matched comparisons.

LIBERO replays initial_states[episode_idx] with a fixed seed, so a 10-rollout probe always scores states
0-9 while baselines and commits score 0-24. EXP-56 measured states 0-9 as 11.5pp easier on 24 merges,
which would make the guard ("best probe on 0-9" > "baseline on 0-24") optimistic by construction. This
uses 50 instrumented episodes of the UNMERGED target per weak target to measure the profile directly,
then rebuilds the guard on matched states and re-counts the harm.
"""
import json, glob, re, sys, pathlib
import numpy as np
from scipy import stats
sys.path.insert(0, "sweep"); sys.path.insert(0, "scripts")
from reinforce_train import scenes as SC, target_baseline
import unified_pool_driver as U
RANK = json.loads(open("results/analysis/probe_selector76_report.txt").read().split("probe rankings: ")[1].strip())
ep = {}
for f in glob.glob("results/merge/e63_base_t*.jsonl"):
    t = int(re.search(r"e63_base_t(\d+)", f).group(1))
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok" and d.get("success_step"):
            ep[t] = [1.0 if s >= 0 else 0.0 for s in d["success_step"]]
if not ep: sys.exit("no e63 records yet")
n = min(len(v) for v in ep.values())
M = np.array([ep[t][:n] for t in sorted(ep)])
print(f"=== EXP-57: per-state difficulty, unmerged targets ({M.shape[0]} targets x {n} states) ===")
per_state = M.mean(0)
print(f"  mean success by state block:")
for lo in range(0, n, 10):
    hi = min(lo + 10, n)
    print(f"    states {lo:>2}-{hi-1:<2}: {100*per_state[lo:hi].mean():>5.1f}%")
print(f"  states 0-9 vs 10-24: {100*per_state[:10].mean():.1f}% vs {100*per_state[10:25].mean():.1f}%"
      f"  -> gap {100*(per_state[:10].mean()-per_state[10:25].mean()):+.1f}pp")
r = stats.pearsonr(np.arange(n), per_state)
print(f"  trend across state index: r={r[0]:+.3f} (p={r[1]:.3f})")
per_t = [(t, 100*(np.mean(ep[t][:10]) - np.mean(ep[t][10:25]))) for t in sorted(ep)]
print(f"  per-target 0-9 minus 10-24: {[(t, round(g)) for t, g in per_t]}")
print(f"  mean {np.mean([g for _, g in per_t]):+.1f}pp (SE {stats.sem([g for _, g in per_t]):.1f}, "
      f"p={stats.ttest_1samp([g for _, g in per_t], 0).pvalue:.4f})")
print(f"\n=== the guard, honest vs current ===")
rows = []
for t in sorted(ep):
    if str(t) not in RANK: continue
    b24 = np.mean(ep[t][:25]); b09 = np.mean(ep[t][:10])
    best = max(s for _, s in RANK[str(t)])
    c = U.get(f"e27_commit_t{t}_top3", SC[t][0])   # U.get takes a single scene, not the list
    if c is None: continue
    rows.append((t, b09, b24, best, c))
if rows:
    for name, fire in (("current: best probe(0-9) > baseline(0-24)", lambda r: r[3] > r[2]),
                       ("honest : best probe(0-9) > baseline(0-9) ", lambda r: r[3] > r[1]),
                       ("honest + 10pp margin                     ", lambda r: r[3] > r[1] + 0.10)):
        d = [100*(r[4]-r[2]) if fire(r) else 0.0 for r in rows]
        nf = sum(1 for r in rows if fire(r)); h = sum(1 for r in rows if fire(r) and r[4] < r[2])
        print(f"  {name}: {np.mean(d):+6.2f}pp   commits {nf}/{len(rows)}   harmed {h}")
    print(f"  (delta always measured against the 25-state baseline, which is how deployment is scored)")
