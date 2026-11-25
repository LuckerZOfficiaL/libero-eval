#!/usr/bin/env python3
"""EXP-103a: spectrum-shape and spectral-TAIL donor scores from "Demystifying Mergeability" (Zhou et al., ICML 2026),
adapted to our residuals. Weights only, CPU.

Per module, residual r_t = D_t - P16(D_t) (same construction as la_pair_features.py), thin SVD r_t = U S V'.
  specov   cos(s_t / sum s_t, s_d / sum s_d) over the 32 singular values      [3]#13 singular-value overlap
  botR     ||V_t,bot' V_d,bot||_F / sqrt(k), k=10 weakest NON-ZERO right vectors  [3]#16 (their best TA/TIES single metric)
  botI     mean sigma^2 of V_t,bot' V_d,bot                                    [3]#18 interaction bottom-k
  topR     same with the 10 strongest (sanity: should match EXP-91 input overlap)
  botL     left (output) version of botR
  smax     sigma_max of r_d (donor-only; Rahamim et al. [4])
Averaged over modules within each band (plain mean, as in [3]); band 0 = all modules.
Output results/analysis/exp103_spec.npz.
"""
import json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
from la_pair_features import load, N
from reinforce_train import group_of
torch.set_num_threads(48)
K = 10; NB = 6
DT = torch.float64

def main():
    t0 = time.time()
    AB = [load(t) for t in range(N)]
    stems = sorted(set.intersection(*[set(x) for x in AB]))
    F = {k: np.zeros((NB, N, N)) for k in ("specov", "botR", "botI", "topR", "botL")}
    F["smax"] = np.zeros((NB, N)); F["rank"] = np.zeros((NB, N)); cnt = np.zeros(NB)
    for mi, stem in enumerate(stems):
        g = group_of(stem) + 1
        b = torch.stack([AB[t][stem][0] for t in range(N)]).to(DT); a = torch.stack([AB[t][stem][1] for t in range(N)]).to(DT)
        dout, din = b.shape[1], a.shape[2]
        Ball = b.permute(1, 0, 2).reshape(dout, N * 16); Aall = a.reshape(N * 16, din)
        Qb, Rb = torch.linalg.qr(Ball); Qa, Ra = torch.linalg.qr(Aall.T)
        Us, sv, Vh = torch.linalg.svd(Rb @ Ra.T)
        U16 = Qb @ Us[:, :16]; V16 = Qa @ Vh[:16].T
        Bp = torch.cat([b, -(U16 @ (U16.T @ b))], 2); Ap = torch.cat([a, (a @ V16) @ V16.T], 1)
        Q, Ra_ = torch.linalg.qr(Ap.transpose(1, 2)); X = Bp @ Ra_.transpose(1, 2)          # r = X Q'
        Ux, S, Wh = torch.linalg.svd(X, full_matrices=False)                                 # r = Ux S (Q W)'
        V = Q @ Wh.transpose(1, 2)                                                            # [N, din, 32] right vectors
        if bool((S[:, 0] <= 1e-12).any()):                                                    # zero residual for some task (untrained module)
            print(f"[sp] skip {stem}: zero residual for tasks {torch.where(S[:, 0] <= 1e-12)[0].tolist()[:8]}", flush=True); continue
        rk = (S > 1e-6 * S[:, :1]).sum(1)                                                     # numerical rank per task
        p = S / S.sum(1, keepdim=True); pn = p / p.norm(dim=1, keepdim=True)
        spec = pn @ pn.T
        def sel(M, bottom):
            idx = [torch.arange(int(r) - K, int(r)) if bottom else torch.arange(K) for r in rk]
            return torch.stack([M[t][:, idx[t]] for t in range(N)])                         # [N, dim, K]
        out = {}
        for nm, M, bottom in (("botR", V, True), ("topR", V, False), ("botL", Ux, True)):
            Z = sel(M, bottom); Zall = Z.permute(1, 0, 2).reshape(Z.shape[1], N * K)
            Gm = (Zall.T @ Zall).reshape(N, K, N, K).permute(0, 2, 1, 3)                     # [t, d, K, K]
            fro2 = Gm.pow(2).sum((2, 3))
            out[nm] = fro2.sqrt() / K ** 0.5
            if nm == "botR":
                out["botI"] = fro2 / K                                                       # mean sigma^2 = ||G||_F^2 / K
        out["specov"] = spec
        for bb in (0, g):
            for k_, v in out.items(): F[k_][bb] += v.numpy()
            F["smax"][bb] += S[:, 0].numpy(); F["rank"][bb] += rk.numpy(); cnt[bb] += 1
        if mi % 25 == 0: print(f"[sp] {mi+1}/{len(stems)} ({dout}x{din}) rank {int(rk.min())}-{int(rk.max())} {(time.time()-t0)/60:.1f} min", flush=True)
    for k_ in F: F[k_] = F[k_] / cnt.reshape((NB,) + (1,) * (F[k_].ndim - 1))
    F["nmod"] = cnt
    np.savez(REPO / "results/analysis/exp103_spec.npz", **F)
    print(f"[sp] DONE {len(stems)} modules in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
