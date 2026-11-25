#!/usr/bin/env python3
"""Extend the affinity study's inputs from 76 to 106 targets (EXP-40, step 1 of the approved plan).

EXP-37 made 30 new experts pool-consistent (capture 0.885 vs 0.001 zero-init), so they can enter the
same integer namespace as the fleet (sweep/plan106.json, indices 76-105). This rebuilds the three
adapter-derived feature blocks over all 106 and merges the labels.

The learning curve measured 0.018pp per training target, so 76 -> 106 predicts +1.31pp -> ~1.85pp
held-out gain. That is a SLOPE test, not a success test: 106 is not the ~140 the curve requires.

Blocks, identical in construction to the 76-target study so the numbers are comparable:
  [W] double-centred Gram row  -- recomputed over 106 (the double-centring removes the shared
      component, which EXP-37 identified as the fleet's warm-start init S16)
  [A] activation signature     -- needs GPU; run activation_probe.py --plan sweep/plan106.json first
  [L] language TF-IDF          -- from the task descriptions
  [S] residual/full effective rank + baseline success
Stages are independent and cached, so a missing GPU block does not block the CPU work.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import safetensors.torch as st
import apply_lora as al

torch.set_num_threads(10)
K = 16
AN = REPO / "results/analysis"
import os
PLAN_FILE = os.environ.get("PLAN", "sweep/plan106.json")          # PLAN=sweep/plan_ext.json for variants
TAG = os.environ.get("TAG", "106")                                  # cache suffix (gram/ranks follow the plan)
FTAG = os.environ.get("FTAG", TAG)                                 # feature-file suffix (varies with block dims)
PLAN = json.load(open(REPO / PLAN_FILE))["adapters"]
AFILE = {a["task_index"]: a["adapter_file"] for a in PLAN}
LANG = {a["task_index"]: a["language"] for a in PLAN}
SCENES = {a["task_index"]: a["scenes"] for a in PLAN}
TASKS = sorted(AFILE)
z = np.load(AN / "loo_soup_factors_t1.npz")
STEMS = [str(s) for s in z["stems"]]
US = {s: torch.from_numpy(z[f"U_{i}"]).float()[:, :K] for i, s in enumerate(STEMS)}
VS = {s: torch.from_numpy(z[f"V_{i}"]).float()[:, :K] for i, s in enumerate(STEMS)}
LAYERS = ["vision", "llm_00", "llm_05", "llm_11", "llm_17", "exp_00", "exp_08", "exp_17", "act_out"]
DIM_W, DIM_A, DIM_L = int(os.environ.get("DIMW", 24)), int(os.environ.get("DIMA", 24)), 16


def factors(t, dtype=torch.float16):
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(AFILE[t])).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.to(dtype), g.b.to(dtype)
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (s, a, b)
    return out


def gram():
    """Pairwise Frobenius inner products of full deltas, same blocked-per-module math as the 76-task
    study, but STREAMED: one safetensors handle per adapter, one module's factors in memory at a time.
    (Loading all adapters at once needed ~16GB for 172 and drove odin to 0 GB free while EXP-32's
    workers were running — never again.)"""
    from safetensors import safe_open
    cache = AN / f"lora_gram{TAG}.npz"
    if cache.exists():
        z_ = np.load(cache)
        print(f"[gram] cache {z_['G'].shape}")
        return z_["G"], [int(x) for x in z_["tasks"]]
    torch.set_num_threads(4)
    N = len(TASKS)
    H = {t: safe_open(AFILE[t], "pt") for t in TASKS}
    keys = set.intersection(*[set(h.keys()) for h in H.values()])
    stems = sorted({k[:-len(".lora_A.weight")] for k in keys if k.endswith(".lora_A.weight")
                    if k[:-len(".lora_A.weight")] + ".lora_B.weight" in keys})
    cfg = {t: json.load(open(pathlib.Path(AFILE[t]).parent / "adapter_config.json")) for t in TASKS}
    r = 16
    G = np.zeros((N, N), dtype=np.float64)
    print(f"[gram] streaming {N} adapters x {len(stems)} modules", flush=True)
    for m, s_ in enumerate(stems):
        Bs, As, sc = [], [], np.zeros(N)
        for i, t in enumerate(TASKS):
            a = H[t].get_tensor(f"{s_}.lora_A.weight").float(); b = H[t].get_tensor(f"{s_}.lora_B.weight").float()
            if b.shape[1] != a.shape[0]:
                a, b = b.T.contiguous(), a.T.contiguous()
            c = cfg[t]; sc[i] = (c.get("lora_alpha", r) / r)
            Bs.append(b); As.append(a)
        B_all = torch.cat(Bs, dim=1); A_all = torch.cat(As, dim=0)
        GB = (B_all.T @ B_all).numpy(); GA = (A_all @ A_all.T).numpy()
        G += np.outer(sc, sc) * (GB * GA).reshape(N, r, N, r).sum(axis=(1, 3))
        if (m + 1) % 100 == 0:
            print(f"[gram]  module {m+1}/{len(stems)}", flush=True)
    np.savez_compressed(cache, G=G, tasks=np.array(TASKS))
    print(f"[gram] saved {cache.name}")
    return G, TASKS


def ranks():
    """Residual and full-delta effective rank (participation ratio), energy-weighted over modules."""
    cache = AN / f"residual_rank{TAG}.json"
    have = json.load(open(cache)) if cache.exists() else {}
    old = json.load(open(AN / "residual_rank.json"))
    if TAG != "106" and (AN / "residual_rank106.json").exists():
        old.update(json.load(open(AN / "residual_rank106.json")))
    have.update({k: v for k, v in old.items() if k not in have})
    good = [s for s in STEMS if float(torch.linalg.norm(US[s], dim=0).min()) > 1e-6][::3]
    pr = lambda sv2: (float(sv2.sum()) ** 2 / float((sv2 ** 2).sum())) if float((sv2 ** 2).sum()) > 0 else np.nan
    todo = [t for t in TASKS if str(t) not in have]
    for n, t in enumerate(todo):
        g = al.extract_lora_groups(st.load_file(AFILE[t]))
        nr = nd = w = 0.0
        for stem in good:
            gg = g[stem]; rk = al.infer_rank(gg.a)
            s = (gg.alpha / rk) if gg.alpha is not None else 1.0
            a, b = gg.a.float(), gg.b.float()
            if b.shape[1] != a.shape[0]:
                a, b = b.T.contiguous(), a.T.contiguous()
            b = b * s
            U = US[stem]; U = U / torch.linalg.norm(U, dim=0); V = VS[stem]
            Cc = (U.T @ b) @ (a @ V)
            Br = torch.cat([b, -(U @ Cc)], 1); Ar = torch.cat([a, V.T], 0)
            qb = torch.linalg.qr(Br, mode="reduced")[1]; qa = torch.linalg.qr(Ar.T, mode="reduced")[1]
            svr = torch.linalg.svdvals(qb @ qa.T) ** 2
            svd = torch.linalg.svdvals(torch.linalg.qr(b, mode="reduced")[1]
                                       @ torch.linalg.qr(a.T, mode="reduced")[1].T) ** 2
            er, ed = pr(svr), pr(svd); e = float(svr.sum())
            if np.isfinite(er) and np.isfinite(ed) and e > 0:
                nr += er * e; nd += ed * e; w += e
        have[str(t)] = [nr / w, nd / w]
        print(f"  ranks t{t}: resid_eff {nr/w:.2f}  ({n+1}/{len(todo)})", flush=True)
        json.dump(have, open(cache, "w"), indent=1)
    return have


def _tagdir(f):
    """map a base record e5X_<h6>_base.jsonl back to the adapter dir it was made for (via the h6 id)."""
    import hashlib
    h = pathlib.Path(f).name.split("_")[1]
    for t in TASKS:
        if t >= 76 and hashlib.md5(AFILE[t].split("/")[-2].encode()).hexdigest()[:6] == h:
            return AFILE[t].split("/")[-2]
    return None


def baselines():
    """Fleet baselines from the standing file; new-expert baselines from EXP-39's records."""
    base = {}
    for ln in open(REPO / "results/baseline_results.jsonl"):
        r = json.loads(ln)
        if r.get("status") == "ok":
            base[r["scene"]] = r["success_rate"]
    out = {}
    for t in TASKS:
        ss = [s for s in SCENES[t] if s in base]
        if ss:
            out[t] = sum(base[s] for s in ss) / len(ss)
    import glob as _g
    for f in _g.glob(str(REPO / "results/merge/e5[134]_*_base.jsonl")):
        for ln in open(f):
            d = json.loads(ln)
            if d.get("status") == "ok":
                for t in TASKS:
                    if SCENES[t] == [d["scene"]] and t >= 76 and AFILE[t].split("/")[-2] == _tagdir(f):
                        out[t] = d["success_rate"]
    return out


def pca(X, d):
    X = np.asarray(X, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    d = min(d, min(X.shape) - 1)
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return U[:, :d] * S[:d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "gram", "ranks", "assemble"])
    a = ap.parse_args()
    if a.stage in ("all", "gram"):
        G, gt = gram()
    if a.stage in ("all", "ranks"):
        rr = ranks()
    if a.stage not in ("all", "assemble"):
        return
    G, gt = gram(); rr = ranks(); base = baselines()
    sigs = {t: (AN / f"act_sigs/task{t}.npz") for t in TASKS}
    missing = [t for t in TASKS if not sigs[t].exists()]
    if missing:
        print(f"[A] MISSING activation signatures for {len(missing)} adapters: {missing[:5]}...")
        print("    run: .venv/bin/python sweep/activation_probe.py --plan sweep/plan106.json "
              f"--tasks {','.join(map(str, missing))} --device cuda:0")
        return
    idx = {t: i for i, t in enumerate(gt)}
    Gc = G - G.mean(0, keepdims=True) - G.mean(1, keepdims=True) + G.mean()
    d = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
    C = Gc / np.outer(d, d)
    keep = [t for t in TASKS if t in base]
    W = np.array([C[idx[t]] for t in keep])
    A = np.array([np.concatenate([np.load(sigs[t])[l].astype(np.float32).mean(0) for l in LAYERS])
                  for t in keep])
    docs = [LANG[t] for t in keep]
    vocab = sorted({w for doc in docs for w in doc.lower().replace(",", " ").split()})
    vi = {w: i for i, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for i, doc in enumerate(docs):
        for w in doc.lower().replace(",", " ").split():
            X[i, vi[w]] += 1
    df = (X > 0).sum(0)
    L = X * np.log(len(docs) / (1 + df))
    S = np.array([[rr[str(t)][0], rr[str(t)][1], base[t]] for t in keep])
    F = np.hstack([pca(W, DIM_W), pca(A, DIM_A), pca(L, DIM_L), (S - S.mean(0)) / (S.std(0) + 1e-8)])
    slices = {"W": (0, DIM_W), "A": (DIM_W, DIM_W + DIM_A),
              "L": (DIM_W + DIM_A, DIM_W + DIM_A + DIM_L), "S": (DIM_W + DIM_A + DIM_L, F.shape[1])}
    np.savez(AN / f"hypernet_feats{FTAG}.npz", F=F, tasks=np.array(keep), slices=json.dumps(slices))
    print(f"[done] hypernet_feats{FTAG}.npz  F={F.shape}  targets={len(keep)} "
          f"(fleet {sum(1 for t in keep if t < 76)}, new {sum(1 for t in keep if t >= 76)})")


if __name__ == "__main__":
    main()
