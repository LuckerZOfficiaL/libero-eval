#!/usr/bin/env python3
"""EXP-11 analyses over activation signatures (results/analysis/act_sigs/).

Part 1 — cheap mergeability signal: per-layer activation distance between a task's solo
policy and its partners' solo policies (on the task's own scenes, and on the full bank),
correlated against the 230-instance real-merge ground truth (all 2/3/4-way average merges).

Part 2 — mechanism: for the 8 instrumented twin merges, per-layer divergence between the
MERGED policy and each parent solo policy on that parent's own scenes — where in depth does
merging change computation, and does late-layer divergence identify the losing twin?

Distances are cosine distances on mean-pooled fp32 activations, averaged over obs rows.
"""
import glob
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
SIGS = REPO / "results" / "analysis" / "act_sigs"
LAYERS = ["vision", "llm_00", "llm_05", "llm_11", "llm_17", "exp_00", "exp_08", "exp_17", "act_out"]

plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
scenes = {a["task_index"]: a["scenes"] for a in plan}
lang = {a["task_index"]: a["language"] for a in plan}
base_sr = {}
for ln in open(REPO / "results/baseline_results.jsonl"):
    d = json.loads(ln)
    if d.get("status") == "ok":
        base_sr[d["scene"]] = d["success_rate"]


def task_rate(sr, t):
    ss = [s for s in scenes[t] if s in sr]
    return sum(sr[s] for s in ss) / len(ss) if ss else None


def load_sig(name):
    z = np.load(SIGS / f"{name}.npz")
    sig = {l: z[l].astype(np.float32) for l in LAYERS}
    return sig, list(z["scene"])


def cosdist(A, B):
    """mean over rows of 1 - cos(A_i, B_i); A,B [n,d]"""
    num = (A * B).sum(1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9
    return float(np.mean(1.0 - num / den))


def main():
    solo, bank_scenes = {}, None
    for t in range(76):
        f = SIGS / f"task{t}.npz"
        if f.exists():
            solo[t], sc = load_sig(f"task{t}")
            bank_scenes = sc
    print(f"[load] {len(solo)}/76 solo signatures")
    rows_of = {t: [i for i, s in enumerate(bank_scenes) if s in scenes[t]] for t in solo}

    # ---------- ground truth: 230 instances from all 2/3/4-way average merges ----------
    inst = []
    for f in sorted(glob.glob(str(REPO / "results/merge/task_*_average.jsonl"))):
        grp = tuple(int(x) for x in re.findall(r"\d+", pathlib.Path(f).stem))
        if len(grp) > 4:
            continue
        sr = {}
        for ln in open(f):
            d = json.loads(ln)
            if d.get("status") == "ok":
                sr[d["scene"]] = d["success_rate"]
        for t in grp:
            m = task_rate(sr, t)
            if m is not None:
                inst.append({"task": t, "grp": grp, "delta": m - task_rate(base_sr, t)})
    print(f"[truth] {len(inst)} merge instances (2/3/4-way)")

    # ---------- Part 1: per-layer solo-vs-partner distance vs transfer ----------
    def sp(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]

    usable = [x for x in inst if x["task"] in solo and all(p in solo for p in x["grp"])]
    dl = np.array([x["delta"] for x in usable])
    print(f"\n=== PART 1: activation distance (task vs partners) vs transfer "
          f"({len(usable)} instances) ===")
    print(f"{'layer':<8}{'own-scenes r':>14}{'(spear)':>9}{'full-bank r':>13}{'(spear)':>9}")
    part1 = {}
    for l in LAYERS:
        own, full = [], []
        for x in usable:
            t, ps = x["task"], [p for p in x["grp"] if p != x["task"]]
            r = rows_of[t]
            own.append(np.mean([cosdist(solo[t][l][r], solo[p][l][r]) for p in ps]))
            full.append(np.mean([cosdist(solo[t][l], solo[p][l]) for p in ps]))
        own, full = np.array(own), np.array(full)
        part1[l] = {"own_r": float(np.corrcoef(own, dl)[0, 1]), "own_s": float(sp(own, dl)),
                    "full_r": float(np.corrcoef(full, dl)[0, 1]), "full_s": float(sp(full, dl))}
        print(f"{l:<8}{part1[l]['own_r']:>+13.3f} {part1[l]['own_s']:>+8.3f}"
              f"{part1[l]['full_r']:>+12.3f} {part1[l]['full_s']:>+8.3f}")

    # ---------- Part 2: merged-vs-solo divergence on twin merges ----------
    CATS = {"26_34": "confusion", "24_69": "confusion", "17_26": "confusion",
            "6_64": "rescue", "19_49": "mild", "44_52": "free", "3_27": "free", "10_29": "free"}
    print(f"\n=== PART 2: merged-vs-solo divergence per layer (parent's own scenes) ===")
    part2 = {}
    for mg, cat in CATS.items():
        f = SIGS / f"merge_{mg}.npz"
        if not f.exists():
            print(f"  merge_{mg}: MISSING")
            continue
        msig, _ = load_sig(f"merge_{mg}")
        grp = [int(x) for x in mg.split("_")]
        # per-parent outcome in this exact pair merge
        rf = REPO / f"results/merge/task_{mg}_average.jsonl"
        sr = {}
        if rf.exists():
            for ln in open(rf):
                d = json.loads(ln)
                if d.get("status") == "ok":
                    sr[d["scene"]] = d["success_rate"]
        for t in grp:
            r = rows_of[t]
            prof = [cosdist(msig[l][r], solo[t][l][r]) for l in LAYERS]
            delta = (task_rate(sr, t) - task_rate(base_sr, t)) if sr else None
            part2[f"{mg}:t{t}"] = {"cat": cat, "delta": delta, "profile": prof}
            ds = f"{delta*100:+.0f}pp" if delta is not None else "n/a"
            print(f"  [{cat:9}] merge {mg} / t{t:<3} ({ds:>6}): " +
                  "  ".join(f"{l.split('_')[0][:3]}{l.split('_')[-1] if '_' in l else ''}={v:.4f}"
                            for l, v in zip(LAYERS, prof)))

    # does action-head divergence identify the loser within each pair?
    print("\n  within-pair: which parent diverges more at act_out vs which parent LOST:")
    hits = tot = 0
    for mg in CATS:
        ks = [k for k in part2 if k.startswith(mg + ":")]
        if len(ks) != 2 or any(part2[k]["delta"] is None for k in ks):
            continue
        a, b = ks
        more_div = a if part2[a]["profile"][-1] > part2[b]["profile"][-1] else b
        loser = a if part2[a]["delta"] < part2[b]["delta"] else b
        tot += 1
        hits += (more_div == loser)
        print(f"    {mg}: more-divergent={more_div.split(':')[1]}  loser={loser.split(':')[1]}"
              f"  {'MATCH' if more_div == loser else 'no'}")
    if tot:
        print(f"  -> action-head divergence identifies the losing twin in {hits}/{tot} pairs")

    json.dump({"part1": part1,
               "part2": {k: {"cat": v["cat"], "delta": v["delta"],
                             "profile": dict(zip(LAYERS, v["profile"]))} for k, v in part2.items()},
               "n_instances": len(usable)},
              open(REPO / "results/analysis/act_analysis.json", "w"), indent=1)
    print(f"\nsaved results/analysis/act_analysis.json")


if __name__ == "__main__":
    main()
