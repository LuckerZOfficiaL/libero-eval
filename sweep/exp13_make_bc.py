#!/usr/bin/env python3
"""EXP-13 mid-chain: pick the mean-best soup dose w* from the completed curve
(w in {0.60, 0.75, 0.85, 0.90} over the 15 weak targets, 10-rollout cells), then
emit stage-b (all weak @ w*, 25 rollouts) and stage-c (all strong @ w*, 10 rollouts) plans.
"""
import glob
import json
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
scenes = {a["task_index"]: a["scenes"] for a in plan}
base = {}
for ln in open(REPO / "results/baseline_results.jsonl"):
    d = json.loads(ln)
    if d.get("status") == "ok":
        base[d["scene"]] = d["success_rate"]


def tr(sr, t):
    ss = [s for s in scenes[t] if s in sr]
    return sum(sr[s] for s in ss) / len(ss) if ss else None


WEAK = json.load(open(REPO / "sweep/weak_targets.json"))
G = json.load(open(REPO / "results/analysis/goal1_full.json"))

deltas = {}                      # (T, w) -> delta
for f in glob.glob(str(REPO / "results/merge/g1w_t*_soup75_w*.jsonl")):
    m = re.match(r"g1w_t(\d+)_soup75_w(\d+)", pathlib.Path(f).stem)
    T, w = int(m.group(1)), int(m.group(2))
    sr = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok":
            sr[d["scene"]] = d["success_rate"]
    v = tr(sr, T)
    if v is not None:
        deltas[(T, w)] = v - tr(base, T)
for T in WEAK:                    # w=75 came from the EXP-12b run
    deltas[(T, 75)] = G["soup75"][str(T)]

means = {}
for w in (60, 75, 85, 90):
    vals = [deltas[(T, w)] for T in WEAK if (T, w) in deltas]
    if len(vals) == len(WEAK):
        means[w] = float(np.mean(vals))
wstar = max(means, key=means.get)
print("dose curve (mean pp on weak):", {w: round(v * 100, 1) for w, v in means.items()},
      "-> w* =", wstar)

tmpl = json.load(open(REPO / "sweep/goal1_full_plan.json"))
meta = {k: tmpl[k] for k in ("base_model", "norm_stats", "n_workers", "gpu_per_worker",
                             "port_per_worker", "results_dir", "videos_dir", "logdir", "save_videos")}
w = wstar / 100.0


def soup_job(name, T):
    others = [t for t in range(76) if t != T]
    return {"name": name, "method": "weighted", "params": {"weights": [w] + [(1 - w) / 75] * 75},
            "adapters": [T] + others, "eval_scenes": scenes[T]}


pb = dict(meta, n_trials=25,
          jobs=[soup_job(f"e13b_t{T}_w{wstar}", T) for T in WEAK])
STRONG = [t for t in range(76) if t not in WEAK]
pc = dict(meta, n_trials=10,
          jobs=[soup_job(f"e13c_t{T}_w{wstar}", T) for T in STRONG])
json.dump(pb, open(REPO / "sweep/exp13_b_plan.json", "w"), indent=1)
json.dump(pc, open(REPO / "sweep/exp13_c_plan.json", "w"), indent=1)
json.dump({"wstar": wstar, "curve": means}, open(REPO / "results/analysis/exp13_dose.json", "w"))
print(f"stage b: {len(pb['jobs'])} jobs @25 | stage c: {len(pc['jobs'])} jobs @10")
