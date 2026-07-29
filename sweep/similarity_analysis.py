#!/usr/bin/env python3
"""LoRA similarity vs merge transfer analysis.

Computes the full pairwise Frobenius inner-product (Gram) matrix over all task
LoRAs via the r x r Gram-trace identity (never materializes an out x in delta):
  <B_iA_i, B_jA_j>_F accumulated per module with blocked matmuls.

From the Gram we get:
  - raw cosine similarity between adapters
  - RESIDUAL cosine: cosine after subtracting the mean adapter direction
    (double-centering the Gram) — removes the shared ~0.8 common component that
    dominates raw cosines, exposing task-specific structure.

Q1: is task similarity detectable in weight space? (same-scene pairs vs others,
    language overlap vs residual cos)
Q2: does similarity to merge partners predict positive/negative transfer?
    (107 task-in-group instances from all completed average merges)

Gram is cached in results/analysis/lora_gram.npz (delete to recompute).
"""
import json
import glob
import math
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, "scripts")
import safetensors.torch as st
import apply_lora as al

torch.set_num_threads(8)
REPO = pathlib.Path(__file__).resolve().parent.parent
AN = REPO / "results" / "analysis"
AN.mkdir(parents=True, exist_ok=True)

plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
scenes = {a["task_index"]: a["scenes"] for a in plan}
lang = {a["task_index"]: a["language"] for a in plan}
afile = {a["task_index"]: a["adapter_file"] for a in plan}
TASKS = sorted(t for t in scenes if t <= 74)
N = len(TASKS)
tpos = {t: i for i, t in enumerate(TASKS)}

# ---------------- Gram matrix (cached) ----------------
cache = AN / "lora_gram.npz"
if cache.exists():
    z = np.load(cache)
    G = z["G"]
    assert list(z["tasks"]) == TASKS
    print(f"[gram] loaded cache {cache.name} ({N}x{N})")
else:
    print(f"[gram] computing {N}x{N} Gram over adapters (blocked, per-module)...")
    AB = {}
    for k, t in enumerate(TASKS):
        state = st.load_file(afile[t])
        d = {}
        for stem, g in al.extract_lora_groups(state).items():
            if g.a is None or g.b is None:
                continue
            r = al.infer_rank(g.a)
            s = (g.alpha / r) if g.alpha is not None else 1.0
            a, b = g.a.to(torch.float16), g.b.to(torch.float16)
            if b.shape[1] != a.shape[0]:
                a, b = b.T.contiguous(), a.T.contiguous()
            d[stem] = (s, a, b)
        AB[t] = d
        if (k + 1) % 15 == 0:
            print(f"[gram]  loaded {k+1}/{N} adapters")
    stems = set.intersection(*[set(AB[t]) for t in TASKS])
    r = 16
    G = np.zeros((N, N), dtype=np.float64)
    svec = np.zeros(N)
    for m, s_ in enumerate(sorted(stems)):
        Bs, As = [], []
        for t in TASKS:
            sc, a, b = AB[t][s_]
            svec[tpos[t]] = sc
            Bs.append(b.to(torch.float32))
            As.append(a.to(torch.float32))
        B_all = torch.cat(Bs, dim=1)               # [out, N*r]
        A_all = torch.cat(As, dim=0)               # [N*r, in]
        GB = (B_all.T @ B_all).numpy()
        GA = (A_all @ A_all.T).numpy()
        M = (GB * GA).reshape(N, r, N, r).sum(axis=(1, 3))
        G += np.outer(svec, svec) * M
        if (m + 1) % 100 == 0:
            print(f"[gram]  {m+1}/{len(stems)} modules")
    np.savez_compressed(cache, G=G, tasks=np.array(TASKS))
    print(f"[gram] saved cache -> {cache}")

norms = np.sqrt(np.diag(G))
C_raw = G / np.outer(norms, norms)
# double-center: <d_i - m, d_j - m> where m = mean adapter
row = G.mean(axis=1, keepdims=True)
Gc = G - row - row.T + G.mean()
cn = np.sqrt(np.clip(np.diag(Gc), 1e-12, None))
C_res = Gc / np.outer(cn, cn)

off = ~np.eye(N, dtype=bool)
print(f"\n=== similarity landscape ===")
print(f"raw cos:      mean {C_raw[off].mean():.3f}  std {C_raw[off].std():.3f}  "
      f"range [{C_raw[off].min():.3f}, {C_raw[off].max():.3f}]")
print(f"residual cos: mean {C_res[off].mean():+.3f}  std {C_res[off].std():.3f}  "
      f"range [{C_res[off].min():+.3f}, {C_res[off].max():+.3f}]")

# ---------------- Q1: is similarity semantic? ----------------
env_re = re.compile(r"^([A-Z_]+SCENE\d+)")
envs = {t: {env_re.match(s).group(1) for s in scenes[t]} for t in TASKS}
STOP = {"the", "a", "an", "and", "of", "it", "its", "to", "on", "in"}
words = {t: {w for w in lang[t].lower().split() if w not in STOP} for t in TASKS}

pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
same_env = np.array([len(envs[TASKS[i]] & envs[TASKS[j]]) > 0 for i, j in pairs])
jac = np.array([len(words[TASKS[i]] & words[TASKS[j]]) / len(words[TASKS[i]] | words[TASKS[j]])
                for i, j in pairs])
res_p = np.array([C_res[i, j] for i, j in pairs])
raw_p = np.array([C_raw[i, j] for i, j in pairs])

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]

def perm_p(x, mask, iters=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = x[mask].mean() - x[~mask].mean()
    cnt = 0
    for _ in range(iters):
        m = rng.permutation(mask)
        if abs(x[m].mean() - x[~m].mean()) >= abs(obs):
            cnt += 1
    return obs, cnt / iters

print(f"\n=== Q1: is task similarity detectable in LoRA space? ===")
d_env, p_env = perm_p(res_p, same_env)
print(f"same physical scene pairs (n={same_env.sum()}) vs rest (n={(~same_env).sum()}):")
print(f"  residual cos: {res_p[same_env].mean():+.4f} vs {res_p[~same_env].mean():+.4f}  "
      f"(diff {d_env:+.4f}, perm p={p_env:.4f})")
d_enr, p_enr = perm_p(raw_p, same_env)
print(f"  raw cos     : {raw_p[same_env].mean():+.4f} vs {raw_p[~same_env].mean():+.4f}  "
      f"(diff {d_enr:+.4f}, perm p={p_enr:.4f})")
print(f"language Jaccard vs residual cos: pearson {np.corrcoef(jac, res_p)[0,1]:+.3f}  "
      f"spearman {spearman(jac, res_p):+.3f}")
print(f"language Jaccard vs raw cos     : pearson {np.corrcoef(jac, raw_p)[0,1]:+.3f}  "
      f"spearman {spearman(jac, raw_p):+.3f}")

idx = np.argsort(res_p)[::-1]
print("\ntop-8 most-similar pairs (residual cos) — semantically related?")
for k in idx[:8]:
    i, j = pairs[k]
    ti, tj = TASKS[i], TASKS[j]
    tag = " [SAME SCENE]" if same_env[k] else ""
    print(f"  {res_p[k]:+.3f}  t{ti} '{lang[ti][:34]}' | t{tj} '{lang[tj][:34]}'{tag}")
print("bottom-4 (most dissimilar):")
for k in idx[-4:]:
    i, j = pairs[k]
    ti, tj = TASKS[i], TASKS[j]
    print(f"  {res_p[k]:+.3f}  t{ti} '{lang[ti][:34]}' | t{tj} '{lang[tj][:34]}'")

# twins sanity check
print("\nnamed pairs (twins etc.):")
for a, b in [(0, 9), (12, 13), (0, 1), (6, 7), (7, 18), (9, 13)]:
    if a in tpos and b in tpos:
        print(f"  t{a} vs t{b}: raw {C_raw[tpos[a], tpos[b]]:.3f}  residual {C_res[tpos[a], tpos[b]]:+.3f}"
              f"   ['{lang[a][:28]}' | '{lang[b][:28]}']")

# ---------------- Q2: similarity vs transfer ----------------
base_sr = {}
for ln in open(REPO / "results/baseline_results.jsonl"):
    d = json.loads(ln)
    if d.get("status") == "ok":
        base_sr[d["scene"]] = d["success_rate"]

def sr_of(f):
    sr = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok":
            sr[d["scene"]] = d["success_rate"]
    return sr

inst = []      # dicts: task, group, delta, res_sim, raw_sim, base, partner_base
groups = []
for f in sorted(glob.glob(str(REPO / "results/merge/*_average.jsonl"))):
    grp = tuple(int(x) for x in re.findall(r"\d+", pathlib.Path(f).stem))
    union = set().union(*[set(scenes[t]) for t in grp])
    sr = sr_of(f)
    if not union <= set(sr):
        continue
    gb = np.mean([base_sr[s] for s in union])
    gm = np.mean([sr[s] for s in union])
    rp = [C_res[tpos[a], tpos[b]] for a in grp for b in grp if b > a]
    groups.append({"grp": grp, "ret": gm / gb, "mean_res": np.mean(rp), "size": len(grp)})
    for t in grp:
        partners = [p for p in grp if p != t]
        m = np.mean([sr[s] for s in scenes[t]])
        b = np.mean([base_sr[s] for s in scenes[t]])
        inst.append({
            "task": t, "grp": grp, "size": len(grp), "delta": m - b, "base": b,
            "partner_base": np.mean([np.mean([base_sr[s] for s in scenes[p]]) for p in partners]),
            "res_sim": np.mean([C_res[tpos[t], tpos[p]] for p in partners]),
            "raw_sim": np.mean([C_raw[tpos[t], tpos[p]] for p in partners]),
        })

print(f"\n=== Q2: does partner similarity predict transfer? ({len(inst)} instances, "
      f"{len(groups)} groups) ===")
dl = np.array([x["delta"] for x in inst])
for name, key in [("mean RESIDUAL cos to partners", "res_sim"),
                  ("mean raw cos to partners", "raw_sim"),
                  ("own baseline (ceiling)", "base"),
                  ("partners' mean baseline", "partner_base")]:
    v = np.array([x[key] for x in inst])
    print(f"  transfer vs {name:<30}: pearson {np.corrcoef(v, dl)[0,1]:+.3f}  "
          f"spearman {spearman(v, dl):+.3f}")

pos = [x for x in inst if x["delta"] >= 0.05]
neg = [x for x in inst if x["delta"] <= -0.05]
neu = [x for x in inst if abs(x["delta"]) < 0.05]
print(f"\n  buckets: positive(+5%+) n={len(pos)}, neutral n={len(neu)}, negative(-5%-) n={len(neg)}")
for nm, xs in [("positive", pos), ("neutral", neu), ("negative", neg)]:
    if xs:
        print(f"    {nm:<9} mean res_sim {np.mean([x['res_sim'] for x in xs]):+.4f}   "
              f"mean own-base {np.mean([x['base'] for x in xs])*100:5.1f}%   "
              f"mean partner-base {np.mean([x['partner_base'] for x in xs])*100:5.1f}%")

gr = np.array([g["ret"] for g in groups])
gs = np.array([g["mean_res"] for g in groups])
print(f"\n  group-level: retention vs mean within-group residual cos: "
      f"pearson {np.corrcoef(gs, gr)[0,1]:+.3f}  spearman {spearman(gs, gr):+.3f}")

json.dump({"instances": [{k: (list(v) if isinstance(v, tuple) else float(v) if isinstance(v, (np.floating, float)) else v)
                          for k, v in x.items()} for x in inst]},
          open(AN / "transfer_instances.json", "w"), indent=1)
print(f"\nsaved instances -> {AN/'transfer_instances.json'}")
