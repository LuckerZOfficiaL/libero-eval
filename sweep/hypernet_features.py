#!/usr/bin/env python3
"""E4 step 1 — per-adapter feature blocks for the affinity hypernetwork.

Produces ONE embedding per task (76 x D) from three views, because the label (behavioural affinity)
is not obviously a function of any single one:

  [W] weight-space : double-centred residual-Gram row. Raw LoRA deltas are useless here (every pair
                     sits at cosine ~0.795, dominated by the shared backbone); double-centring the
                     Gram removes that common component and leaves the residual similarities.
  [A] behaviour    : solo activation signature on the FIXED 180-observation bank (9 hook depths),
                     mean-pooled over observations. Policy-independent inputs, so adapters are
                     directly comparable. NB: activation *distance* failed as a monotone selector;
                     the signature is kept here as a FEATURE, which is a different claim.
  [L] language     : TF-IDF over the task description + SVD. Cheap, dependency-light, and the piece
                     genuinely analogous to Text-to-LoRA (which conditions on text alone).
  [S] scalars      : residual effective rank, full-delta effective rank, baseline success.

Each block is standardised and PCA'd separately, then concatenated, so no single block can dominate
purely by dimensionality. Output: results/analysis/hypernet_feats.npz
"""
import json
import pathlib

import numpy as np

REPO = pathlib.Path("/home/lzhou/libero-eval")
SIGS = REPO / "results/analysis/act_sigs"
LAYERS = ["vision", "llm_00", "llm_05", "llm_11", "llm_17", "exp_00", "exp_08", "exp_17", "act_out"]
DIM_W, DIM_A, DIM_L = 24, 24, 16


def pca(X, d, seed=0):
    """Standardise then project to d dims (or fewer if rank-limited)."""
    X = np.asarray(X, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    d = min(d, min(X.shape) - 1)
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return (U[:, :d] * S[:d])


def block_weight(tasks):
    z = np.load(REPO / "results/analysis/lora_gram.npz")
    G, gt = z["G"].astype(np.float64), [int(x) for x in z["tasks"]]
    # double-centre: strips the shared backbone that makes every raw cosine ~0.795
    Gc = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
    d = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
    C = Gc / np.outer(d, d)                       # residual cosine matrix
    idx = {t: i for i, t in enumerate(gt)}
    rows = []
    for t in tasks:                               # duplicates (t7=t18, t29=t62) fall back to their twin
        if t in idx:
            rows.append(C[idx[t]])
        else:
            rows.append(C[idx[{7: 18, 18: 7, 29: 62, 62: 29}.get(t, gt[0])]])
    return np.array(rows)


def block_act(tasks):
    rows = []
    for t in tasks:
        z = np.load(SIGS / f"task{t}.npz")
        rows.append(np.concatenate([z[l].astype(np.float32).mean(0) for l in LAYERS]))
    return np.array(rows)


def block_lang(tasks):
    plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
    lang = {a["task_index"]: a["language"] for a in plan}
    docs = [lang[t] for t in tasks]
    vocab = sorted({w for d in docs for w in d.lower().replace(",", " ").split()})
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w in d.lower().replace(",", " ").split():
            X[i, vi[w]] += 1
    df = (X > 0).sum(0)
    return X * np.log(len(docs) / (1 + df))       # tf-idf; SVD happens in pca()


def block_scalar(tasks):
    rr = json.load(open(REPO / "results/analysis/residual_rank.json"))
    plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
    scenes = {a["task_index"]: a["scenes"] for a in plan}
    base = {}
    for ln in open(REPO / "results/baseline_results.jsonl"):
        r = json.loads(ln)
        if r.get("status") == "ok":
            base[r["scene"]] = r["success_rate"]
    out = []
    for t in tasks:
        ss = [s for s in scenes[t] if s in base]
        b = sum(base[s] for s in ss) / len(ss)
        re_, fe_ = rr[str(t)]
        out.append([re_, fe_, b])
    return np.array(out)


def main():
    tasks = list(range(76))
    W, A, L, S = block_weight(tasks), block_act(tasks), block_lang(tasks), block_scalar(tasks)
    print(f"raw blocks  W{W.shape}  A{A.shape}  L{L.shape}  S{S.shape}")
    F = np.hstack([pca(W, DIM_W), pca(A, DIM_A), pca(L, DIM_L),
                   (S - S.mean(0)) / (S.std(0) + 1e-8)])
    slices = {"W": (0, DIM_W), "A": (DIM_W, DIM_W + DIM_A),
              "L": (DIM_W + DIM_A, DIM_W + DIM_A + DIM_L), "S": (DIM_W + DIM_A + DIM_L, F.shape[1])}
    np.savez(REPO / "results/analysis/hypernet_feats.npz", F=F, tasks=np.array(tasks),
             slices=json.dumps(slices))
    print(f"wrote hypernet_feats.npz  F={F.shape}  blocks={slices}")


if __name__ == "__main__":
    main()
