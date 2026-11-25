#!/usr/bin/env python3
"""EXP-89 stage 2 — behavioural check of a LOCAL shared component, donor-free.

Swap arm with a group soup instead of the pool:  W = S_G + (D_t - P_G(D_t))
  S_G = top-16 two-sided SVD of the MEAN FULL DELTA of group G (not containing t or its duplicate)
  P_G = two-sided projection onto those 16 directions; the target keeps everything else of its own adapter.
Arms (G of size 8):  nn8  = its 8 nearest residual-cosine neighbours (same groups as linalg_structure.py)
                     rnd8 = 8 random tasks (seeded) -- separates "local" from "built from fewer tasks"
Reference arms already measured: k=16 pool swap (e94_t{t}_k16) and the baseline.
15 weak fleet targets, 25 rollouts, records e95_t{t}_{arm}.
"""
import json, os, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
import safetensors.torch as st
from reinforce_train import scenes as SC
from subset_probe import TARGETS
if os.environ.get("LS_SLICE"):
    _a, _b = os.environ["LS_SLICE"].split(":"); TARGETS = TARGETS[int(_a or 0):int(_b) if _b else None]

DUP = {7: 18, 18: 7, 29: 62, 62: 29}
C = np.load(REPO / "results/analysis/residual_cosines.npz")["C"]
ARMS = ["nn8", "rnd8"]
def group(t, arm):
    ex = {t, DUP.get(t, -1)}
    if arm == "nn8":
        return [int(g) for g in np.argsort(-C[t]) if g not in ex][:8]
    rng = np.random.default_rng(8900 + t)
    return sorted(int(x) for x in rng.choice([g for g in range(76) if g not in ex], 8, replace=False))

def build(name, t, G, k=16):
    tf = U.fold(t); folds = [U.fold(g) for g in G]; tensors = {}
    for stem in U.stems:
        if not all(stem in f for f in folds): continue
        Bg = torch.cat([f[stem][0] for f in folds], 1).double() / len(G)
        Ag = torch.cat([f[stem][1] for f in folds], 0).double()
        Qb, Rb = torch.linalg.qr(Bg); Qa, Ra = torch.linalg.qr(Ag.T)
        Us, sv, Vh = torch.linalg.svd(Rb @ Ra.T)
        Un = (Qb @ Us[:, :k]); V = (Qa @ Vh[:k].T)
        Usig = (Un * sv[:k]).float(); Un, V = Un.float(), V.float()
        Bb, Ab = [Usig], [V.T]
        if stem in tf:
            b, a = tf[stem]; M = Un.T @ b; N = a @ V
            Bb += [b, -(Un @ M)]; Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    U._ensure_space(1.0); st.save_file(tensors, str(p)); return str(p)

def main():
    U._clear_orphaned_builds()
    todo = [(t, a) for t in TARGETS for a in ARMS if U.get(f"e95_t{t}_{a}", SC[t][0]) is None]
    print(f"[ls] {len(todo)} cells", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, arm in todo[i:i + U.CHUNK]:
            G = group(t, arm); p = build(f"ls_{t}_{arm}", t, G); built.append(p)
            print(f"[ls] t{t} {arm}: group {G}", flush=True)
            jobs.append({"name": f"e95_t{t}_{arm}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [SC[t][0]]})
        U.run(jobs, U.COMMIT_R, "libero_90", U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[ls] done", flush=True)

if __name__ == "__main__":
    main()
