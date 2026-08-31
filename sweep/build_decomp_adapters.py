#!/usr/bin/env python3
"""Phase B builder: synthetic adapters from the shared/residual decomposition.

For target t (with saved LOO-soup factors): per module, with orthonormal soup dirs
(u_j, v_j) and soup top-k matrix S_k = sum_j sigma_j u_j v_j^T, build three variants:

  interchange : S_k + (D_t - P_k(D_t))   — OTHERS' shared + own residual
  residual    : D_t - P_k(D_t)           — own residual only
  shared      : S_k                      — others' shared only

Each variant is written as a synthetic LoRA safetensors (lora_A/lora_B factors, no
alpha keys -> scale falls back to 1.0), rank <= k+32, loadable by merge_loras.py.
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import safetensors.torch as st
import apply_lora as al

REPO = pathlib.Path(__file__).resolve().parent.parent
OUTD = REPO / "results" / "analysis" / "decomp_adapters"
OUTD.mkdir(parents=True, exist_ok=True)
K = 16
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}


def build_for(t):
    z = np.load(REPO / f"results/analysis/loo_soup_factors_t{t}.npz")
    stems = [str(s) for s in z["stems"]]
    Us = {s: torch.from_numpy(z[f"U_{i}"]).to(torch.float32) for i, s in enumerate(stems)}
    Vs = {s: torch.from_numpy(z[f"V_{i}"]).to(torch.float32) for i, s in enumerate(stems)}
    groups = al.extract_lora_groups(st.load_file(afile[t]))
    out = {"interchange": {}, "residual": {}, "shared": {}}
    for stem in stems:
        g = groups[stem]
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.to(torch.float32), g.b.to(torch.float32)
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        b = b * s                                     # fold scale: D_t = b @ a
        Usig = Us[stem][:, :K]                        # u_j * sigma_j
        V = Vs[stem][:, :K]                           # orthonormal v_j
        sig = torch.linalg.norm(Usig, dim=0)
        U = Usig / sig.clamp(min=1e-12)               # orthonormal u_j
        # projection blocks: P(D) = U (U^T b a V) V^T
        M2 = U.T @ b                                  # [k, r]
        N2 = a @ V                                    # [r, k]
        # variant factor blocks (delta = B_hat @ A_hat)
        def pack(Bblocks, Ablocks):
            return torch.cat(Bblocks, 1).contiguous(), torch.cat(Ablocks, 0).contiguous()
        Bi, Ai = pack([Usig, b, -(U @ M2)], [V.T, a, N2 @ V.T])       # S_k + D - P(D)
        Br, Ar = pack([b, -(U @ M2)], [a, N2 @ V.T])                  # D - P(D)
        Bs_, As_ = pack([Usig], [V.T])                                # S_k
        for name, (B_, A_) in (("interchange", (Bi, Ai)), ("residual", (Br, Ar)), ("shared", (Bs_, As_))):
            out[name][f"{stem}.lora_A.weight"] = A_.to(torch.float32)
            out[name][f"{stem}.lora_B.weight"] = B_.to(torch.float32)
    paths = {}
    for name, tensors in out.items():
        p = OUTD / f"t{t}_{name}_k{K}.safetensors"
        st.save_file(tensors, str(p))
        paths[name] = str(p)
        print(f"  t{t} {name:<12} rank={next(iter(tensors.values())).shape[0] if 'lora_A' in next(iter(tensors)) else '?'} -> {p.name}")
    return paths


if __name__ == "__main__":
    allp = {}
    for t in (1, 6):
        print(f"[build] t{t}")
        allp[str(t)] = build_for(t)
    json.dump(allp, open(OUTD / "paths.json", "w"), indent=1)
    print("done")
