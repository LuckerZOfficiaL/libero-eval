# 11-05 diagnostic: the converted externals share a common mode (pairwise cos ~0.10 vs 0.035 for WP on demos, all orders).
# Does the averaged in-context update (EXP-145, which fixes order 0) mostly cancel that common mode? If so, removing it
# training-free could replace the in-context round.
import os, sys, pathlib
os.environ.setdefault("E112_DEV", "cpu"); os.environ["E144_SEQS"] = "0"; os.environ["E145_VARS"] = "ctxr,ctx"; os.environ["E145_LAMS"] = ""
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import torch; torch.set_num_threads(8)
import exp145_incontext as Q
S, E, U = Q.S, Q.E, Q.U
def priv(i):
    f = U.fold(E.LIB[i]["key"]); out = {}
    for s_, (B, A) in f.items():
        if s_ not in U.USIG: continue
        Un = U.USIG[s_] / torch.linalg.norm(U.USIG[s_], dim=0).clamp(min=1e-12); out[s_] = B - Un @ (Un.T @ B)
    return out
def flat(d, ks): return torch.cat([d[k].float().flatten() for k in ks])
NEW = Q.NEW; wp = S.W.sequence(0)[1][:S.T]
P = {"conv": [priv(S.CV[i]) for i in NEW], "wp": [priv(j) for j in wp], "ctx": [priv(Q.CX["ctx"][i]) for i in NEW], "ctxr": [priv(Q.CX["ctxr"][i]) for i in NEW]}
ks = sorted(set.intersection(*[set(d) for v in P.values() for d in v]))
F = {k: torch.stack([flat(d, ks) for d in v]) for k, v in P.items()}
def cos(a, b): return float(a @ b / a.norm() / b.norm())
m = F["conv"].mean(0); mw = F["wp"].mean(0)
print(f"|mean conv| {m.norm():.2f} (|conv_i| ~{F['conv'].norm(dim=1).mean():.2f}); |mean WP| {mw.norm():.2f}; cos(mean conv, mean WP) {cos(m, mw):.3f}")
for v in ("ctx", "ctxr"):
    d = (F[v] - F["conv"]).mean(0)
    print(f"{v}: |mean update| {d.norm():.2f}; cos(mean update, mean conv) {cos(d, m):+.3f}; coef <d,m>/|m|^2 {float(d @ m / (m @ m)):+.3f}; "
          f"cos(mean update, mean WP - mean conv) {cos(d, mw - m):+.3f}; per-external cos(update_i, conv_i) "
          + " ".join(f"{cos(a - b, b):+.2f}" for a, b in zip(F[v], F["conv"])))
# common-mode share of each converted private part, and how much the update removes from the merge's external sum
R = F["conv"] - m
print(f"common-mode energy share of the 8-sum: conv {float((8*m).norm()**2 / (F['conv'].sum(0)).norm()**2):.2f}, WP {float((8*mw).norm()**2 / (F['wp'].sum(0)).norm()**2):.2f}")
