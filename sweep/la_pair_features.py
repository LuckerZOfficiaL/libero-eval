#!/usr/bin/env python3
"""EXP-91 stage 1 — linear-algebra donor scores for EVERY (target, donor) pair of the 76 fleet adapters.

Per module, residuals in operator form r = D - P16(D), factored r_t = X_t Q_t' (Q_t orthonormal row basis)
and r_t = P_t Y_t' (P_t orthonormal column basis). Accumulated per BAND (vision, LLM early, LLM late,
action expert, action head), energy-weighted (numerators summed; divide by summed denominators):
  ip     <r_t, r_d>                       -> residual cosine, and |ip| for module-level sign agreement
  OL     ||P_t' P_d||^2 / 32              -> overlap of residual OUTPUT (column) subspaces
  OR     ||Q_t' Q_d||^2 / 32              -> overlap of residual INPUT (row) subspaces
  DinT2  ||P_t' r_d Q_t||^2               -> donor residual energy inside the target's two-sided subspace
  DinTL  ||P_t' r_d||^2                   -> ... inside the target's output subspace
  DinTR  ||r_d Q_t||^2                    -> ... inside the target's input subspace
  ipD    <D_t, D_d> (full deltas)         -> full-delta cosine
  rn2, dn2 per task per band              -> denominators
Output results/analysis/la_pair_features.npz. Evaluation: sweep/la_pair_eval.py.
"""
import json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
import apply_lora as al
from reinforce_train import group_of
torch.set_num_threads(48)
N, NB = 76, 5
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}

def load(t):
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(afile[t])).items():
        if g.a is None or g.b is None: continue
        r = al.infer_rank(g.a); s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]: a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = ((b * s).contiguous(), a.contiguous())
    return out

def blocks(M, n=N, r=32):     # [n*r, n*r] -> [n, n, r, r]
    return M.reshape(n, r, n, r).permute(0, 2, 1, 3)

def main():
    t0 = time.time()
    AB = [load(t) for t in range(N)]
    stems = sorted(set.intersection(*[set(x) for x in AB]))
    F = {k: np.zeros((NB, N, N)) for k in ("ip", "absip", "OL", "OR", "DinT2", "DinTL", "DinTR", "ipD")}
    F["rn2"] = np.zeros((NB, N)); F["dn2"] = np.zeros((NB, N))
    DT = torch.float64
    for mi, stem in enumerate(stems):
        g = group_of(stem)
        b = torch.stack([AB[t][stem][0] for t in range(N)]).to(dev, DT); a = torch.stack([AB[t][stem][1] for t in range(N)]).to(dev, DT)
        dout, din = b.shape[1], a.shape[2]
        Ball = b.permute(1, 0, 2).reshape(dout, N * 16); Aall = a.reshape(N * 16, din)
        Qb, Rb = torch.linalg.qr(Ball); Qa, Ra = torch.linalg.qr(Aall.T)
        Us, sv, Vh = torch.linalg.svd(Rb @ Ra.T)
        U16 = Qb @ Us[:, :16]; V16 = Qa @ Vh[:16].T
        Ub = U16.T @ b; aV = a @ V16
        Bp = torch.cat([b, -(U16 @ Ub)], 2); Ap = torch.cat([a, aV @ V16.T], 1)
        Q, Ra_ = torch.linalg.qr(Ap.transpose(1, 2)); X = Bp @ Ra_.transpose(1, 2)     # r = X Q'
        P, _ = torch.linalg.qr(Bp)                                                    # column basis of r
        Xall = X.permute(1, 0, 2).reshape(dout, N * 32); Pall = P.permute(1, 0, 2).reshape(dout, N * 32)
        Qall = Q.permute(1, 0, 2).reshape(din, N * 32)
        GX = blocks(Xall.T @ Xall); PP = blocks(Pall.T @ Pall); QQ = blocks(Qall.T @ Qall); PX = blocks(Pall.T @ Xall)
        GXd = GX[torch.arange(N), torch.arange(N)]                                     # [N, 32, 32] X_d' X_d
        ip = (GX * QQ).sum((2, 3))                                                      # <r_t, r_d> = sum(X_t'X_d * Q_t'Q_d)
        rn2 = ip.diagonal()
        OL = PP.pow(2).sum((2, 3)) / 32; OR = QQ.pow(2).sum((2, 3)) / 32
        # donor d inside target t: P_t' r_d Q_t = (P_t' X_d)(Q_d' Q_t) = PX[t,d] @ QQ[d,t]
        QQdt = QQ.transpose(0, 1)                                                       # [t, d] -> Q_d' Q_t
        DinT2 = (PX @ QQdt).pow(2).sum((2, 3))
        DinTL = PX.pow(2).sum((2, 3))
        # ||r_d Q_t||^2 = ||X_d (Q_d' Q_t)||^2 = tr((Q_d'Q_t)' X_d'X_d (Q_d'Q_t))
        DinTR = ((GXd[None] @ QQdt) * QQdt).sum((2, 3))
        GB = (Ball.T @ Ball).reshape(N, 16, N, 16).permute(0, 2, 1, 3); GA = (Aall @ Aall.T).reshape(N, 16, N, 16).permute(0, 2, 1, 3)
        ipD = (GB * GA).sum((2, 3)); dn2 = ipD.diagonal()
        for k, v in (("ip", ip), ("absip", ip.abs()), ("OL", OL), ("OR", OR), ("DinT2", DinT2), ("DinTL", DinTL), ("DinTR", DinTR), ("ipD", ipD)):
            F[k][g] += v.cpu().numpy()
        F["rn2"][g] += rn2.cpu().numpy(); F["dn2"][g] += dn2.cpu().numpy()
        if mi % 25 == 0: print(f"[lf] {mi+1}/{len(stems)} ({dout}x{din}) {(time.time()-t0)/60:.1f} min", flush=True)
    F["nmod"] = np.array([sum(1 for s in stems if group_of(s) == g) for g in range(NB)])
    np.savez(REPO / "results/analysis/la_pair_features.npz", **F)
    print(f"[lf] DONE {len(stems)} modules in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
