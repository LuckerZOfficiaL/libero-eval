#!/usr/bin/env python3
"""EXP-92 / EXP-93 — linear-algebra variants of HOW to merge, on the 15 weak fleet targets, 25 rollouts/cell.

EXP-93 (no donors): own-residual dose
    b75 / b125 / b150 :  W = S16 + beta * r_t            (beta = 1 is the e94 k16 pool swap, already measured)
EXP-92 (fixed donors = the target's probe-committed top-3 from EXP-27, same donors in every arm):
    std  :  W = S16 + r_t + sum_d (c_d/3) r_d                               c_d = ref/||r_d|| (the operator)
    in2  :  donor residual replaced by its part INSIDE the target's residual subspace (two-sided projection
            onto the column and row spaces of r_t, per module), rescaled to the same total norm as r_d
    top4 :  donor residual truncated to its top-4 singular directions per module, rescaled to the same norm
All transforms are energy-matched (a global per-donor scalar), so arms differ in STRUCTURE, not dose.
Records e96_t{t}_{arm} (EXP-93) and e97_t{t}_{arm} (EXP-92). Env: LA_ARMS=comma list, LA_SLICE=a:b.
"""
import json, os, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
import safetensors.torch as st
from reinforce_train import scenes as SC
from subset_probe import TARGETS
if os.environ.get("LA_SLICE"):
    _a, _b = os.environ["LA_SLICE"].split(":"); TARGETS = TARGETS[int(_a or 0):int(_b) if _b else None]
ARMS = os.environ.get("LA_ARMS", "b75,b125,b150,std,in2,top4").split(",")
BETA = {"b75": 0.75, "b125": 1.25, "b150": 1.5}
PREFIX = {**{a: "e96" for a in BETA}, "std": "e97", "in2": "e97", "top2": "e97", "top4": "e97", "top8": "e97"}
if os.environ.get("LA_PREFIX"):                     # replicate run on fresh rollouts under a new record prefix
    PREFIX = {a: os.environ["LA_PREFIX"] for a in PREFIX}
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))

def top3(t):
    pr = sorted(((s, int(p)) for p, s in U.E27[str(t)]), key=lambda z: -z[0])
    return [p for _, p in pr[:3]]

def ops(stem):
    Usig = U.USIG[stem]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = U.VS[stem]
    return Usig, Un, V

def resid_factors(key, stem, Un, V):
    b, a = U.fold(key)[stem]; M = Un.T @ b; N = a @ V
    return torch.cat([b, -(Un @ M)], 1), torch.cat([a, N @ V.T], 0)            # r = Bp @ Ap (rank <= 32)

def transform(arm, Bp, Ap, Pt, Qt):
    if arm == "in2":                                       # P_t P_t' r Q_t Q_t'
        return Pt @ ((Pt.T @ Bp) @ (Ap @ Qt)), Qt.T
    if arm.startswith("top"):                              # spectral truncation of the donor residual
        k = int(arm[3:])
        Qb, Rb = torch.linalg.qr(Bp); Qa, Ra = torch.linalg.qr(Ap.T)
        Us, sv, Vh = torch.linalg.svd(Rb @ Ra.T)
        return (Qb @ Us[:, :k]) * sv[:k], (Qa @ Vh[:k].T).T
    return Bp, Ap

def fro2(B, A):
    return float(((B.T @ B) * (A @ A.T)).sum())

def build(name, t, arm):
    tensors = {}; tf = U.fold(t)
    if arm in BETA:
        beta = BETA[arm]
        for stem in U.stems:
            Usig, Un, V = ops(stem); Bb, Ab = [Usig], [V.T]
            if stem in tf:
                Bp, Ap = resid_factors(t, stem, Un, V); Bb.append(beta * Bp); Ab.append(Ap)
            tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous(); tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    else:
        donors = top3(t)
        # pass 1: per-donor norm of the transformed residual, for energy matching
        scale = {}
        for d in donors:
            n0 = n1 = 0.0
            for stem in U.stems:
                if stem not in U.fold(d) or stem not in tf: continue
                Usig, Un, V = ops(stem)
                Bp, Ap = resid_factors(d, stem, Un, V); n0 += fro2(Bp, Ap)
                if arm != "std":
                    Bt, At = resid_factors(t, stem, Un, V)
                    Pt, _ = torch.linalg.qr(Bt); Qt, _ = torch.linalg.qr(At.T)
                    B2, A2 = transform(arm, Bp, Ap, Pt, Qt); n1 += fro2(B2, A2)
            c = REF / (n0 ** 0.5) / 3
            scale[d] = c * ((n0 / n1) ** 0.5 if arm != "std" else 1.0)
        for stem in U.stems:
            Usig, Un, V = ops(stem); Bb, Ab = [Usig], [V.T]
            if stem in tf:
                Bt, At = resid_factors(t, stem, Un, V); Bb.append(Bt); Ab.append(At)
                Pt, _ = torch.linalg.qr(Bt); Qt, _ = torch.linalg.qr(At.T)
                for d in donors:
                    if stem not in U.fold(d): continue
                    Bp, Ap = resid_factors(d, stem, Un, V)
                    B2, A2 = transform(arm, Bp, Ap, Pt, Qt)
                    Bb.append(scale[d] * B2); Ab.append(A2)
            tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous(); tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    U._ensure_space(1.7); st.save_file(tensors, str(p)); return str(p)

def main():
    U._clear_orphaned_builds()
    todo = [(t, a) for a in ARMS for t in TARGETS if U.get(f"{PREFIX[a]}_t{t}_{a}", SC[t][0]) is None]
    print(f"[la] arms {ARMS} targets {TARGETS}: {len(todo)} cells; ref {REF:.3f}", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, arm in todo[i:i + U.CHUNK]:
            p = build(f"la_{t}_{arm}", t, arm); built.append(p)
            jobs.append({"name": f"{PREFIX[arm]}_t{t}_{arm}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [SC[t][0]]})
        U.run(jobs, U.COMMIT_R, "libero_90", U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[la] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[la] done", flush=True)

if __name__ == "__main__":
    main()
