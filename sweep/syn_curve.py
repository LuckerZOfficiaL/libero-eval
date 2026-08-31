#!/usr/bin/env python3
"""GO/NO-GO — do SYNTHETIC targets extend the learning curve like real ones?

Adds the completed e32 synthetics to the binary-objective training set (holdout stays 10 REAL targets)
and asks whether gain-vs-random rises with the number of synthetics at the real-target slope
(~0.018pp/target). Features for synthetics are computed in the SAME projection as the real tasks:
the residual Gram is bilinear, so <r_syn, r_j> = Gc[t,j] + 0.5*Gc[p,j], and the PCA basis fitted on
the 76 real rows is reused for synthetic rows. Blocks WLS (activation block excluded: it carries no
signal for real tasks and cannot be computed for synthetics without GPU).
"""
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path("/home/lzhou/libero-eval")
sys.path.insert(0, str(REPO / "sweep"))
import hypernet as H
import hypernet_features as HF

# ---------------------------------------------------------------- synthetic labels
SPEC = json.load(open(REPO / "sweep/syn_spec.json"))
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
scenes = {a["task_index"]: a["scenes"] for a in plan}


def jrate(job, t):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return None
    sr = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok":
            sr[d["scene"]] = d["success_rate"]
    ss = [s for s in scenes[t] if s in sr]
    return sum(sr[s] for s in ss) / len(ss) if len(ss) == len(scenes[t]) else None


syn_rows = []          # (spec, solo, [(donor, affinity_pp)])
for s in SPEC:
    solo = jrate(f"e32_syn{s['id']}_solo", s["task"])
    if solo is None:
        continue
    labs = []
    for d in s["cands"]:
        v = jrate(f"e32_syn{s['id']}_p{d}", s["task"])
        if v is not None:
            labs.append((d, (v - solo) * 100))
    if len(labs) >= 6:
        syn_rows.append((s, solo, labs))
print(f"complete synthetics: {len(syn_rows)}")

# ---------------------------------------------------------------- features (WLS, shared projection)
tasks = list(range(76))
Wraw, Lraw, Sraw = HF.block_weight(tasks), HF.block_lang(tasks), HF.block_scalar(tasks)


def fit_proj(X, d):
    X = np.asarray(X, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    d = min(d, min(Z.shape) - 1)
    return (U[:, :d] * S[:d]), (mu, sd, Z.mean(0), Vt[:d])


def project(x, P):
    mu, sd, zmu, Vt = P
    return (((x - mu) / sd) - zmu) @ Vt.T


Wf, Wp = fit_proj(Wraw, HF.DIM_W)
Lf, Lp = fit_proj(Lraw, HF.DIM_L)
smu, ssd = Sraw.mean(0), Sraw.std(0) + 1e-8
Sf = (Sraw - smu) / ssd
F_real = np.hstack([Wf, Lf, Sf])

# synthetic W rows via Gram bilinearity
z = np.load(REPO / "results/analysis/lora_gram.npz")
G, gt = z["G"].astype(np.float64), [int(x) for x in z["tasks"]]
Gc = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
dg = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
idx = {t: i for i, t in enumerate(gt)}
TWIN = {7: 18, 18: 7, 29: 62, 62: 29}
def gi(t):
    return idx[t] if t in idx else idx[TWIN.get(t, gt[0])]
rr = json.load(open(REPO / "results/analysis/residual_rank.json"))

F_syn = []
for s, solo, labs in syn_rows:
    t, p = gi(s["task"]), gi(s["pert"])
    g = Gc[t] + 0.5 * Gc[p]
    n2 = Gc[t, t] + 0.25 * Gc[p, p] + Gc[t, p]
    wrow = g / (np.sqrt(max(n2, 1e-12)) * dg)
    lrow = Lraw[s["task"]]
    srow = np.array([rr[str(s["task"])][0], rr[str(s["task"])][1], solo])
    F_syn.append(np.concatenate([project(wrow, Wp), project(lrow, Lp), (srow - smu) / ssd]))
F_all = torch.tensor(np.vstack([F_real] + ([np.vstack(F_syn)] if F_syn else [])), dtype=torch.float32)

# ---------------------------------------------------------------- groups
F76, groups_real, baseline = H.load(str(REPO / "results/analysis/pair_labels.json"))
# rebuild real groups against the SAME WLS feature matrix (indices 0..75 align)
syn_groups = []
for k, (s, solo, labs) in enumerate(syn_rows):
    o = np.random.default_rng(9500 + k).permutation(len(labs))
    syn_groups.append((76 + k, [labs[i][0] for i in o],
                       np.array([labs[i][1] for i in o], np.float32),
                       np.full(len(labs), 10.0, np.float32)))

# ---------------------------------------------------------------- curve
from scipy import stats
SEEDS = 10
levels = [0, len(syn_groups) // 2, len(syn_groups)]
print(f"levels of added synthetics: {levels}\n")
out = {}
for K in levels:
    gains = []
    for seed in range(SEEDS):
        tr, te, ho = H.split(groups_real, baseline, seed=seed)
        if K:
            sel = np.random.default_rng(3000 + seed).choice(len(syn_groups), K, replace=False)
            tr = tr + [syn_groups[i] for i in sel]
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = H.AffinityNet(F_all.shape[1], d_emb=32, p_drop=0.1)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(150):
            net.train()
            for i in np.random.permutation(len(tr)):
                T, cand, aff, lw = tr[i]
                sc = net(F_all[T].expand(len(cand), -1), F_all[cand])
                loss = H.binary_loss(sc, aff, lw)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()
        def score(T, cand):
            with torch.no_grad():
                return net(F_all[T].expand(len(cand), -1), F_all[cand]).numpy()
        m = H.evaluate(score, te)
        gains.append(m["topn"] - m["random"])
    g = np.array(gains)
    out[K] = g
    t = stats.ttest_1samp(g, 0)
    print(f"  +{K:<3} synthetics: gain vs random {g.mean():+.2f}pp  SE {stats.sem(g):.2f}  "
          f"p={t.pvalue:.3f}   per-seed {np.round(g,1).tolist()}")
k1, k2 = levels[0], levels[-1]
d = out[k2] - out[k1]
t = stats.ttest_rel(out[k2], out[k1])
exp = 0.018 * k2
print(f"\n  adding {k2} synthetics: {d.mean():+.2f}pp (SE {stats.sem(d):.2f}, paired p={t.pvalue:.3f})")
print(f"  real-target slope predicts: +{exp:.2f}pp if synthetics were equivalent to real targets")
print("SYN CURVE DONE")
