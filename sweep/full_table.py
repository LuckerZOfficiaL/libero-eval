#!/usr/bin/env python3
"""Rebuild the pairwise merge results table FRESH from results/merge/*.jsonl.

Per (job, scene) we take the latest status=='ok' entry (parse_fail/error ignored).
A pair-cell is the mean over that job's ok scenes; it is 'incomplete' when fewer
scenes are ok than the pair actually has, so a half-run never hides as done.

Baseline column = the UPPER BOUND: for pair task_a_b, the mean success rate of the
two parent LoRAs each evaluated on their own task's scene(s) (from baseline_results).

By default prints only configs complete across ALL pairs (the fair comparison);
pass --all to also show partial/single-pair configs.
"""
import glob
import json
import pathlib
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
PLAN = REPO / "sweep" / "plan.json"
BASE = REPO / "results" / "baseline_results.jsonl"
MERGE = REPO / "results" / "merge"
SHOW_ALL = "--all" in sys.argv

scenes = {a["task_index"]: a["scenes"] for a in json.load(open(PLAN))["adapters"]}
base_sr = {}
for ln in open(BASE):
    d = json.loads(ln)
    if d.get("status") == "ok":
        base_sr[d["scene"]] = d["success_rate"]


def pretty(method):
    if method == "average":
        return "average"
    if method.startswith("task_arithmetic"):
        return "ta " + method.split("_")[-1].replace("lambda", "λ")
    if method.startswith("ties"):
        return "ties " + method.split("_")[-1].replace("lambda", "λ")
    return method


cells = defaultdict(dict)               # (pair, method) -> {scene: sr}
for f in sorted(glob.glob(str(MERGE / "*.jsonl"))):
    toks = pathlib.Path(f).stem.split("_")
    pair, method = f"{toks[1]}_{toks[2]}", pretty("_".join(toks[3:]))
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok":
            cells[(pair, method)][d["scene"]] = d["success_rate"]

pairs = sorted({p for p, _ in cells}, key=lambda s: tuple(int(x) for x in s.split("_")))
all_cols = ["average", "ta λ0.3", "ta λ0.5", "ta λ0.7", "ta λ1.0", "ties λ1.0", "ties λ0.7"]


def pscenes(pair):
    a, b = (int(x) for x in pair.split("_"))
    return sorted(set(scenes[a]) | set(scenes[b]))


# per-column list of complete-pair means (a cell counts only when all its scenes are ok)
complete = defaultdict(dict)
for pair in pairs:
    exp = pscenes(pair)
    for c in all_cols:
        d = cells.get((pair, c))
        if d and len(d) == len(exp):
            complete[c][pair] = sum(d.values()) / len(d)

cols = all_cols if SHOW_ALL else [c for c in all_cols if len(complete[c]) == len(pairs)]


def cell(pair, c):
    d = cells.get((pair, c))
    if not d:
        return "–"
    exp = pscenes(pair)
    s = "*" if len(d) < len(exp) else ""
    return f"{sum(d.values())/len(d)*100:.0f}%{s}"


base_pair = {p: sum(base_sr[s] for s in pscenes(p)) / len(pscenes(p)) for p in pairs}
hdr = ["pair", "base↑"] + cols
w = 11
print("  ".join(f"{h:>{w}}" for h in hdr))
print("-" * ((w + 2) * len(hdr)))
for pair in pairs:
    row = [f"task_{pair}", f"{base_pair[pair]*100:.0f}%"] + [cell(pair, c) for c in cols]
    print("  ".join(f"{x:>{w}}" for x in row))
print("-" * ((w + 2) * len(hdr)))
mean = ["MEAN", f"{sum(base_pair.values())/len(pairs)*100:.1f}%"] + [
    (f"{sum(complete[c].values())/len(complete[c])*100:.1f}%" if len(complete[c]) == len(pairs) else "n/a")
    for c in cols]
print("  ".join(f"{x:>{w}}" for x in mean))
if not SHOW_ALL:
    print(f"\nColumns shown = configs complete on all {len(pairs)} pairs. base↑ = upper bound "
          "(both parent LoRAs on their own task). Use --all to include partial configs.")
