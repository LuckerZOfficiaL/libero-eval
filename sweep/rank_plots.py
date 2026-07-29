#!/usr/bin/env python3
"""Rank-position histograms of merged success rate, by merge size.

Within each merge group (size N), each task's merged success rate = mean over that
task's own scene(s). Tasks are ranked best->worst inside the group; then for each
rank position we average across all groups of that size. Bar 1 = best-preserved
task on average, bar N = most-fragile. One chart per size (2/3/4-way).

Output: results/figs/rank_{2,3,4}way.png and results/figs/rank_all.png
Rerun anytime (e.g. after the full triplet sweep) to refresh.
"""
import json
import glob
import pathlib
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parent.parent
scenes = {a["task_index"]: a["scenes"] for a in json.load(open(REPO / "sweep/plan.json"))["adapters"]}
OUT = REPO / "results" / "figs"
OUT.mkdir(parents=True, exist_ok=True)


def merged_sr(f):
    sr = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok":
            sr[d["scene"]] = d["success_rate"]
    return sr


# size -> list of ranked per-task success arrays (one per complete group)
by_size = defaultdict(list)
for f in sorted(glob.glob(str(REPO / "results/merge/*_average.jsonl"))):
    grp = tuple(int(x) for x in re.findall(r"\d+", pathlib.Path(f).stem))
    union = set().union(*[set(scenes[t]) for t in grp])
    sr = merged_sr(f)
    if not union <= set(sr):
        continue                                   # incomplete group -> skip
    per_task = [np.mean([sr[s] for s in scenes[t]]) for t in grp]
    by_size[len(grp)].append(sorted(per_task, reverse=True))   # best -> worst


def panel(ax, size):
    groups = by_size.get(size, [])
    M = len(groups)
    arr = np.array(groups)                          # [M, size]
    means = arr.mean(axis=0) * 100
    stds = arr.std(axis=0) * 100
    ranks = np.arange(1, size + 1)
    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.15, size))   # green(best) -> red(fragile)
    ax.bar(ranks, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    for r, m in zip(ranks, means):
        ax.text(r, m + 2.5, f"{m:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(f"{size}-way merges  (n={M} groups)", fontsize=12, fontweight="bold")
    ax.set_xlabel("task rank within merge")
    ax.set_xticks(ranks)
    labels = ["1\n(best)"] + [str(r) for r in ranks[1:-1]] + [f"{size}\n(fragile)"]
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("mean merged success rate (%)")
    ax.grid(axis="y", alpha=0.3)
    return means


# combined figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
tbl = {}
for ax, size in zip(axes, (2, 3, 4)):
    tbl[size] = panel(ax, size)
fig.suptitle("Merged success rate by within-group task rank (best-preserved → most-fragile)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "rank_all.png", dpi=150)

# individual figures
for size in (2, 3, 4):
    f1, a1 = plt.subplots(figsize=(1.6 + 1.4 * size, 4.6))
    panel(a1, size)
    f1.tight_layout()
    f1.savefig(OUT / f"rank_{size}way.png", dpi=150)

# numeric table to stdout
print("rank-position mean success rate (%), averaged across groups:")
for size in (2, 3, 4):
    vals = "  ".join(f"r{i+1}={v:.0f}%" for i, v in enumerate(tbl[size]))
    print(f"  {size}-way (n={len(by_size[size])}): {vals}")
print(f"\nsaved: {OUT}/rank_all.png  and rank_2way/3way/4way.png")
