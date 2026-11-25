# 10-27 EXP-145 diagnostic: did the 8 in-context updates move in a SHARED direction (Jacobi overshoot when summed)?
import os, sys, pathlib
os.environ.setdefault("E112_DEV", "cpu"); os.environ.setdefault("E144_SEQS", "0")
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import torch; torch.set_num_threads(8)
import exp145_incontext as Q
import exp144_selfconvert as S
import exp112_bench as E
import unified_pool_driver as U
def priv(i):
    f = U.fold(E.LIB[i]["key"]); out = {}
    for s_, (B, A) in f.items():
        if s_ not in U.USIG: continue
        Un = U.USIG[s_] / torch.linalg.norm(U.USIG[s_], dim=0).clamp(min=1e-12); out[s_] = B - Un @ (Un.T @ B)
    return out
def flat(d, ks): return torch.cat([d[k].flatten() for k in ks])
conv = {i: priv(S.CV[i]) for i in Q.NEW}; ks = sorted(set.intersection(*[set(v) for v in conv.values()]))
for v in Q.VARS:
    D = []
    for i in Q.NEW:
        c = flat(priv(Q.CX[v][i]), ks); b = flat(conv[i], ks); D.append(c - b)
        print(f"{v} {E.LIB[i]['tid']}  |B_perp| conv {b.norm():.2f} -> {c.norm():.2f}   |update| {(c - b).norm():.2f}")
    M = torch.stack(D); M = M / M.norm(dim=1, keepdim=True); G = M @ M.T
    off = G[~torch.eye(len(D), dtype=bool)]
    S_ = torch.stack(D).sum(0)
    print(f"{v}: pairwise cosine of the 8 updates mean {off.mean():.3f} (min {off.min():.3f} max {off.max():.3f}); "
          f"|sum of updates| {S_.norm():.2f} vs sqrt(sum |update|^2) {torch.stack(D).norm():.2f} (independent) / sum |update| {sum(d.norm() for d in D):.2f} (aligned)")
