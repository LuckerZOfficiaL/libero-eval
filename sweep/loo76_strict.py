#!/usr/bin/env python3
"""Strict 76-fold LOO capture (paper-grade version of full76_capture).
For EACH task t: SVD of the soup of the other 75 (per module), capture of t under its
top-k. Parallel over worker shards: usage: loo76_strict.py <shard> <nshards>."""
import json, pathlib, sys
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import safetensors.torch as st
import apply_lora as al
torch.set_num_threads(4)
REPO = pathlib.Path(__file__).resolve().parent.parent
KS = [1, 2, 4, 8, 16, 32, 64]
SHARD, NSHARDS = int(sys.argv[1]), int(sys.argv[2])
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}

def load_AB(t):
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(afile[t])).items():
        if g.a is None or g.b is None: continue
        r = al.infer_rank(g.a); s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.to(torch.float32), g.b.to(torch.float32)
        if b.shape[1] != a.shape[0]: a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (s, a, b)
    return out

AB = {t: load_AB(t) for t in range(76)}
stems = sorted(set.intersection(*[set(AB[t]) for t in AB]))
targets = [t for t in range(76) if t % NSHARDS == SHARD]
out = {}
for t_out in targets:
    others = [t for t in range(76) if t != t_out]
    capv = np.zeros(len(KS)); totv = 0.0
    for stem in stems:
        Bs, As = [], []
        for i in others:
            s, a, b = AB[i][stem]
            Bs.append(b * (s / len(others))); As.append(a)
        Bc = torch.cat(Bs, 1); Ac = torch.cat(As, 0)
        Qb, Rb = torch.linalg.qr(Bc, mode="reduced")
        Qa, Ra = torch.linalg.qr(Ac.T, mode="reduced")
        Us, sv, Vst = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
        U = Qb @ Us; V = Qa @ Vst.T
        s, a, b = AB[t_out][stem]
        totv += s * s * float(((b.T @ b) * (a @ a.T).T).sum())
        UB = U.T @ b; AV = a @ V
        for ki, k in enumerate(KS):
            m_ = UB[:k] @ AV[:, :k]
            capv[ki] += s * s * float((m_ ** 2).sum())
    out[str(t_out)] = (capv / totv).tolist()
    print(f"[shard {SHARD}] t{t_out} done: c16={out[str(t_out)][KS.index(16)]:.3f}", flush=True)
json.dump({"ks": KS, "curves": out},
          open(REPO / f"results/analysis/loo76_strict_shard{SHARD}.json", "w"), indent=1)
print(f"[shard {SHARD}] COMPLETE ({len(out)} targets)")
