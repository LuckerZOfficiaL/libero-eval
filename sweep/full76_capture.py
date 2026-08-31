#!/usr/bin/env python3
"""All-76 capture analysis under the FULL-76 soup's top-k directions (Phase A, complete version).
Justified by the measured LOO~full equivalence (held-out vs held-in gap <=2pp, 1/75 membership).
Outputs per-task capture at k in {1..64} + distribution stats + correlation with transfer history."""
import json, pathlib, sys
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import safetensors.torch as st
import apply_lora as al
torch.set_num_threads(12)
REPO = pathlib.Path(__file__).resolve().parent.parent
KS = [1, 2, 4, 8, 16, 32, 64]
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
cap = {t: np.zeros(len(KS)) for t in range(76)}
tot = {t: 0.0 for t in range(76)}
for mi, stem in enumerate(stems):
    Bs, As = [], []
    for i in range(76):
        s, a, b = AB[i][stem]
        Bs.append(b * (s / 76)); As.append(a)
    Bc = torch.cat(Bs, 1); Ac = torch.cat(As, 0)
    Qb, Rb = torch.linalg.qr(Bc, mode="reduced")
    Qa, Ra = torch.linalg.qr(Ac.T, mode="reduced")
    Us, sv, Vst = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    U = Qb @ Us; V = Qa @ Vst.T
    for t in range(76):
        s, a, b = AB[t][stem]
        tot[t] += s * s * float(((b.T @ b) * (a @ a.T).T).sum())
        UB = U.T @ b; AV = a @ V
        for ki, k in enumerate(KS):
            m_ = UB[:k] @ AV[:, :k]
            cap[t][ki] += s * s * float((m_ ** 2).sum())
    if (mi + 1) % 100 == 0: print(f"  {mi+1}/{len(stems)} modules", flush=True)

curves = {t: (cap[t] / tot[t]).tolist() for t in range(76)}
json.dump({"ks": KS, "curves": {str(t): c for t, c in curves.items()}},
          open(REPO / "results/analysis/full76_capture.json", "w"), indent=1)
c16 = np.array([curves[t][KS.index(16)] for t in range(76)])
print(f"\ncapture@k=16 over ALL 76 adapters: mean {c16.mean():.3f}  sd {c16.std():.3f}  "
      f"min {c16.min():.3f} (t{int(np.argmin(c16))})  max {c16.max():.3f} (t{int(np.argmax(c16))})")
