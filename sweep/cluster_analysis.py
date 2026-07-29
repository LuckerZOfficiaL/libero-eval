#!/usr/bin/env python3
"""Cluster analysis of LoRA residual space.

Uses the cached 75x75 Gram (results/analysis/lora_gram.npz). Raw adapters are
~0.795-cosine to each other (one shared direction), so clustering happens in the
RESIDUAL space: double-center the Gram (= subtract the mean adapter), giving both
  - classical MDS coordinates (eigendecomposition of the centered Gram), and
  - residual cosine distances for average-linkage hierarchical clustering.
Duplicate adapters (t18=t7, t29 kept / t62 dropped) are excluded.

Reports: eigen-spectrum (how much structure the residual space has), clusters at
several k with language/scene labels, semantic enrichment (verb / object / scene
env), and where fragile vs anchor tasks fall.

Figure: results/figs/lora_residual_mds.png
"""
import json
import pathlib
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parent.parent
z = np.load(REPO / "results/analysis/lora_gram.npz")
G_full, TASKS_full = z["G"], list(z["tasks"])
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
lang = {a["task_index"]: a["language"] for a in plan}
scenes = {a["task_index"]: a["scenes"] for a in plan}

DROP = {18, 62}                       # byte-identical duplicates of 7 and 29
keep = [i for i, t in enumerate(TASKS_full) if t not in DROP]
TASKS = [TASKS_full[i] for i in keep]
G = G_full[np.ix_(keep, keep)]
N = len(TASKS)

# residual space: double-center the Gram
row = G.mean(axis=1, keepdims=True)
Gc = G - row - row.T + G.mean()
cn = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
C = Gc / np.outer(cn, cn)             # residual cosine
D = np.clip(1.0 - C, 0, None)
np.fill_diagonal(D, 0.0)

# ---- eigen-spectrum: how much structure? ----
w, V = np.linalg.eigh(Gc)
w = w[::-1]; V = V[:, ::-1]
tot = w[w > 0].sum()
print("=== residual-space eigen-spectrum (top 10, % of residual variance) ===")
print("  " + "  ".join(f"{100*x/tot:.1f}%" for x in w[:10]))
print(f"  dims to reach 50% var: {int(np.searchsorted(np.cumsum(w[w>0])/tot, .5))+1}, "
      f"80%: {int(np.searchsorted(np.cumsum(w[w>0])/tot, .8))+1}  (of {N})")
XY = V[:, :2] * np.sqrt(np.clip(w[:2], 0, None))     # MDS coords

# ---- fragility history from transfer instances ----
inst = json.load(open(REPO / "results/analysis/transfer_instances.json"))["instances"]
hist = defaultdict(list)
for x in inst:
    hist[x["task"]].append(x["delta"])
avg_delta = {t: float(np.mean(v)) for t, v in hist.items()}

def frag_tag(t):
    if t not in avg_delta:
        return "unseen"
    d = avg_delta[t]
    return "fragile" if d <= -0.10 else ("robust" if d >= 0.0 else "mid")

# ---- semantic features ----
env_re = re.compile(r"^([A-Z_]+SCENE\d+)")
env_of = {t: sorted({env_re.match(s).group(1) for s in scenes[t]}) for t in TASKS}
VERBS = ["put", "pick", "stack", "open", "close", "turn"]
OBJECTS = ["bowl", "book", "mug", "pan", "pot", "sauce", "soup", "ketchup", "milk",
           "butter", "pudding", "cheese", "wine", "bottle", "drawer", "microwave",
           "stove", "cabinet", "plate", "basket", "tray", "caddy", "moka", "salad"]
def feats(t):
    words = lang[t].lower()
    return ([v for v in VERBS if words.startswith(v) or f" {v} " in words],
            [o for o in OBJECTS if o in words])

# ---- hierarchical clustering ----
L = linkage(squareform(D, checks=False), method="average")
print("\n=== clusters (average linkage on residual cosine distance) ===")
for k in (5, 8):
    lab = fcluster(L, t=k, criterion="maxclust")
    print(f"\n--- k={k} ---")
    for c in sorted(set(lab)):
        idx = [i for i in range(N) if lab[i] == c]
        ts = [TASKS[i] for i in idx]
        verbs = Counter(); objs = Counter(); envs = Counter(); frs = Counter()
        for t in ts:
            v, o = feats(t)
            verbs.update(v); objs.update(o); frs[frag_tag(t)] += 1
            envs.update(e.split("_SCENE")[0] for e in env_of[t])
        head = ", ".join(f"{w}x{c_}" for w, c_ in verbs.most_common(2))
        obj = ", ".join(f"{w}x{c_}" for w, c_ in objs.most_common(3))
        env = ", ".join(f"{w}x{c_}" for w, c_ in envs.most_common(2))
        fr = "/".join(f"{k_}:{v_}" for k_, v_ in frs.items() if k_ != "unseen")
        print(f"  C{c} (n={len(ts)}): verbs[{head}] objs[{obj}] envs[{env}] frag[{fr}]")
        print(f"      tasks: {ts}")
        for t in ts[:4]:
            print(f"        t{t}: {lang[t][:60]}")

# best-friend view: nearest neighbor in residual space
print("\n=== nearest neighbor (residual cos) for notable tasks ===")
for t in [0, 6, 12, 15, 2, 13, 24, 69]:
    if t not in TASKS: continue
    i = TASKS.index(t)
    j = int(np.argsort(-C[i])[1])
    print(f"  t{t:<3}({frag_tag(t):7}) -> t{TASKS[j]:<3} cos {C[i,j]:+.3f}   "
          f"'{lang[t][:32]}' -> '{lang[TASKS[j]][:32]}'")

# ---- figure: MDS scatter, clusters at k=5, fragility as ring ----
lab5 = fcluster(L, t=5, criterion="maxclust")
PAL = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]
fig, ax = plt.subplots(figsize=(9.5, 7.5))
for c in sorted(set(lab5)):
    idx = [i for i in range(N) if lab5[i] == c]
    ax.scatter(XY[idx, 0], XY[idx, 1], s=64, c=PAL[(c - 1) % len(PAL)],
               label=f"C{c} (n={len(idx)})", alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)
for i, t in enumerate(TASKS):
    tag = frag_tag(t)
    if tag == "fragile":
        ax.scatter([XY[i, 0]], [XY[i, 1]], s=150, facecolors="none",
                   edgecolors="#D55E00", linewidths=2.0, zorder=4)
    if tag == "fragile" or t in (2, 13, 24, 69, 26, 34):
        ax.annotate(f"t{t}", (XY[i, 0], XY[i, 1]), textcoords="offset points",
                    xytext=(6, 5), fontsize=8.5, color="#333333")
ax.scatter([], [], s=150, facecolors="none", edgecolors="#D55E00", linewidths=2.0,
           label="fragile (avg Δ ≤ −10pp)")
ax.set_xlabel(f"MDS-1  ({100*w[0]/tot:.0f}% of residual var)")
ax.set_ylabel(f"MDS-2  ({100*w[1]/tot:.0f}%)")
ax.set_title("LoRA adapters in residual weight space (classical MDS, duplicates removed)",
             fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="best")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(REPO / "results/figs/lora_residual_mds.png", dpi=150)
print(f"\nsaved figure -> results/figs/lora_residual_mds.png")
