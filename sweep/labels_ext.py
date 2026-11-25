#!/usr/bin/env python3
"""Merge every probe-derived pair label we own into one file keyed by plan_ext task indices.

fleet (634, pair_labels.json) + e51 (30 S16-init experts) + e52 (6 short) + e53 (10 seed variants)
+ e54 (50 seed variants). Keys '<target>_<donor>' with targets at their plan_ext.json indices, so the
affinity study can train on variants (indices >= 106) while holding out only the original 106.
"""
import json, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parent.parent
plan = json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]
by_dir = {a["adapter_file"].split("/")[-2]: a["task_index"] for a in plan if a["task_index"] >= 76}
merged = json.load(open(REPO / "results/analysis/pair_labels.json"))
n_fleet = len(merged); added = {}
for rep in ("newpool_probe_report.txt", "newpool_short_report.txt", "newpool_seeds_report.txt", "newpool_var50_report.txt"):
    f = REPO / "results/analysis" / rep
    if not f.exists():
        print(f"  (missing {rep})"); continue
    lab = json.loads(f.read_text().split("labels: ", 1)[1].strip())
    k = 0
    for name, v in lab.items():
        if name not in by_dir:
            continue
        t = by_dir[name]
        for p, s in v["probes"].items():
            merged[f"{t}_{p}"] = {"aff": (s - v["base"]) * 100, "w": 10}; k += 1
    added[rep] = k
out = REPO / "results/analysis/pair_labels_ext.json"
json.dump(merged, open(out, "w"), indent=0)
tg = {int(x.split("_")[0]) for x in merged}
print(f"pair_labels_ext.json: {len(merged)} labels over {len(tg)} targets (fleet {n_fleet}; added {added})")
print(f"  targets >=106 (variants, train-only): {sum(1 for t in tg if t >= 106)}")
