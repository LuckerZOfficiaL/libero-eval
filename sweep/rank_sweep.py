#!/usr/bin/env python3
"""EXP-90 stage 2 — per-task rank of the shared component, donor-free (no partner draw, no lottery).

Swap arm at rank k:  W = S_k + (D_t - P_k(D_t))
  S_k  = the pool's shared component at rank k (top-k of the LOO-soup, same factors the operator uses)
  P_k  = two-sided projection onto its k directions; so the target keeps EVERYTHING of its own adapter
         except its own top-k shared coordinates, which are replaced by the pool's.
k = 0 is the target itself (baseline, pooled from 3 independent 25-rollout measurements).
k = 16 is the operator's S16 swap (the Phase-B interchange). k in {2, 4, 8, 12, 16}.

The 15 weak fleet targets, 25 rollouts per cell, records e94_t{t}_k{k}. Stage 2b (--confirm) re-runs each
target's best k and k=16 on fresh rollouts (records e94c_*), because picking a per-task best of 5 noisy
cells has a winner's curse (the project has been bitten by this 3 times).
"""
import json, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
import safetensors.torch as st
from reinforce_train import scenes as SC, target_baseline
from subset_probe import TARGETS
import os
if os.environ.get("RS_SLICE"):                  # "a:b" -> split the 15 targets across machines
    _a, _b = os.environ["RS_SLICE"].split(":"); TARGETS = TARGETS[int(_a or 0):int(_b) if _b else None]

KS = [2, 4, 8, 12, 16]
z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
fstems = [str(s) for s in z["stems"]]
USIG64 = {s: torch.from_numpy(z[f"U_{i}"]).float() for i, s in enumerate(fstems)}
V64 = {s: torch.from_numpy(z[f"V_{i}"]).float() for i, s in enumerate(fstems)}

def build_swap(name, t, k):
    tf = U.fold(t); tensors = {}
    for stem in fstems:
        Usig = USIG64[stem][:, :k]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = V64[stem][:, :k]
        Bb, Ab = [Usig], [V.T]
        if stem in tf:
            b, a = tf[stem]; M = Un.T @ b; N = a @ V
            Bb += [b, -(Un @ M)]; Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    U._ensure_space(1.0); st.save_file(tensors, str(p)); return str(p)

def base(t):
    v = [x for x in (target_baseline(t), U.get(f"e44_base_f{t}", SC[t][0]), U.get(f"e92_base_f{t}", SC[t][0])) if x is not None]
    return float(np.mean(v)), len(v)

def run_cells(cells, prefix):
    todo = [(t, k) for t, k in cells if U.get(f"{prefix}_t{t}_k{k}", SC[t][0]) is None]
    print(f"[rs] {prefix}: {len(todo)} cells to run", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, k in todo[i:i + U.CHUNK]:
            p = build_swap(f"rs_{prefix}_{t}_{k}", t, k); built.append(p)
            jobs.append({"name": f"{prefix}_t{t}_k{k}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [SC[t][0]]})
        U.run(jobs, U.COMMIT_R, "libero_90", U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[rs] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)

def main():
    U._clear_orphaned_builds()
    if "--confirm" in sys.argv:
        cells = []
        for t in TARGETS:
            sc = {k: U.get(f"e94_t{t}_k{k}", SC[t][0]) for k in KS}
            if any(v is None for v in sc.values()): continue
            kb = max(KS, key=lambda k: sc[k])
            cells += [(t, kb)] + ([(t, 16)] if kb != 16 else [])
        run_cells(cells, "e94c")
    else:
        run_cells([(t, k) for t in TARGETS for k in KS], "e94")
    print("[rs] done", flush=True)

if __name__ == "__main__":
    main()
