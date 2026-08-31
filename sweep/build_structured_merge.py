#!/usr/bin/env python3
"""Phase C builder: the structured merge operator.

  W_structured = S_16  +  sum_i ( D_i - P_16(D_i) )

One copy of the shared backbone (soup top-16 per module) plus every member task's
residual at FULL strength (no 1/N dilution). Shared factors: t1's LOO-soup factors
(valid for any group not containing t1 — Phase A measured LOO~full <=2pp and Phase B
proved the shared component is interchangeable).

Groups: twin pair 24:69 (classic confusion), dissimilar pair 5:33 (known-clean),
mixed triplet 12:13:14. Plain-average references already exist for all three.
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
K = 16
GROUPS = [(24, 69), (5, 33), (12, 13, 14)]
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}

z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
stems = [str(s) for s in z["stems"]]
US = {s: torch.from_numpy(z[f"U_{i}"]).to(torch.float32)[:, :K] for i, s in enumerate(stems)}
VS = {s: torch.from_numpy(z[f"V_{i}"]).to(torch.float32)[:, :K] for i, s in enumerate(stems)}


def load_fold(t):
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(afile[t])).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.to(torch.float32), g.b.to(torch.float32)
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)                    # D = b @ a with scale folded
    return out


def main():
    for grp in GROUPS:
        assert 1 not in grp, "shared factors come from t1's LOO soup"
        ABs = {t: load_fold(t) for t in grp}
        tensors = {}
        for stem in stems:
            Usig = US[stem]
            sig = torch.linalg.norm(Usig, dim=0)
            U = Usig / sig.clamp(min=1e-12)
            V = VS[stem]
            Bblocks, Ablocks = [Usig], [V.T]
            for t in grp:
                b, a = ABs[t][stem]
                M2 = U.T @ b
                N2 = a @ V
                Bblocks += [b, -(U @ M2)]
                Ablocks += [a, N2 @ V.T]
            tensors[f"{stem}.lora_A.weight"] = torch.cat(Ablocks, 0).contiguous()
            tensors[f"{stem}.lora_B.weight"] = torch.cat(Bblocks, 1).contiguous()
        name = "structured_" + "_".join(map(str, grp)) + f"_k{K}"
        st.save_file(tensors, str(OUTD / f"{name}.safetensors"))
        rank = tensors[f"{stems[0]}.lora_A.weight"].shape[0]
        print(f"  {name}: rank {rank} per module")


if __name__ == "__main__":
    main()
