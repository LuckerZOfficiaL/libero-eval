#!/usr/bin/env python
"""Summarize merge results vs the single-task baseline (retention).

For every merged (pair, scene) result, show the merged success rate next to the scene's single-task
baseline (EXP-2), and the delta (retention). Aggregate per pair and overall. Reads all per-job files
in results/merge/ (one per task_<a>_<b>_<method>).

Usage: python sweep/merge_summary.py [results/merge  OR  a single results/merge/<job>.jsonl]
"""
import glob, json, os, sys, statistics
from collections import defaultdict

REPO = "/home/lzhou/libero-eval"
arg = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/results/merge"
files = glob.glob(f"{arg}/*.jsonl") if os.path.isdir(arg) else [arg]

# baseline: scene -> single-task success
base = {}
try:
    for l in open(f"{REPO}/results/baseline_results.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok":
            base[r["scene"]] = r["success_rate"]
except FileNotFoundError:
    pass

# merged results (dedupe by (job, scene), keep last)
merged = {}
for fp in files:
    for l in open(fp):
        try:
            r = json.loads(l)
            if r.get("status") == "ok":
                merged[(r["job"], r["scene"])] = r
        except Exception:
            pass

by_job = defaultdict(list)
for (job, scene), r in merged.items():
    by_job[job].append(r)

print(f"{'merged':>7} {'baseline':>8} {'delta':>7}  job / scene")
print("-" * 90)
all_deltas = []
for job in sorted(by_job):
    rs = sorted(by_job[job], key=lambda r: r["scene"])
    for r in rs:
        b = base.get(r["scene"])
        d = (r["success_rate"] - b) if b is not None else None
        if d is not None:
            all_deltas.append(d)
        bs = f"{b*100:5.0f}%" if b is not None else "   ?  "
        ds = f"{d*100:+5.0f}%" if d is not None else "   ?  "
        print(f"{r['success_rate']*100:6.0f}% {bs:>8} {ds:>7}  {job} / {r['scene'][:40]}")
    mm = statistics.mean(x["success_rate"] for x in rs) * 100
    bb = [base[x["scene"]] for x in rs if x["scene"] in base]
    bmean = statistics.mean(bb) * 100 if bb else float("nan")
    print(f"   -> pair mean: merged {mm:.0f}%  vs baseline {bmean:.0f}%  (n_trials={rs[0].get('n_trials')})\n")

if all_deltas:
    print(f"OVERALL retention (merged - baseline): mean {statistics.mean(all_deltas)*100:+.1f}%  "
          f"over {len(all_deltas)} (pair,scene) evals; worst {min(all_deltas)*100:+.0f}%")
