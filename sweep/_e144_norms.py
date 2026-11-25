# 11-05 EXP-144 diagnostic: why does the converted merge keep externals in order 0 (44.5%) but not orders 1-3 (8-22%)?
# per external: |B_perp| of the converted adapter vs the WP-on-demos adapter, and their cosine
import os, sys, pathlib
os.environ.setdefault("E112_DEV", "cpu"); os.environ["E144_SEQS"] = "0,1,2,3"
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import torch; torch.set_num_threads(8)
import exp144_selfconvert as S
import exp112_bench as E
import unified_pool_driver as U
def priv(i):
    f = U.fold(E.LIB[i]["key"]); out = {}
    for s_, (B, A) in f.items():
        if s_ not in U.USIG: continue
        Un = U.USIG[s_] / torch.linalg.norm(U.USIG[s_], dim=0).clamp(min=1e-12); out[s_] = B - Un @ (Un.T @ B)
    return out
def flat(d, ks): return torch.cat([d[k].float().flatten() for k in ks])
for s_ in (0, 1, 2, 3):
    new = S.C.sequence(s_)[1][:S.T]; wp = S.W.sequence(s_)[1][:S.T]; nc, nw = [], []
    for i, j in zip(new, wp):
        c, w = priv(S.CV[i]), priv(j); ks = sorted(set(c) & set(w)); c, w = flat(c, ks), flat(w, ks)
        nc.append(c.norm()); nw.append(w.norm())
        print(f"s{s_} {E.LIB[i]['tid']}  |B_perp| conv {c.norm():.2f}  WP-demos {w.norm():.2f}  cos {(c @ w / c.norm() / w.norm()):.3f}", flush=True)
    C = torch.stack([flat(priv(S.CV[i]), ks) for i in new]); C = C / C.norm(dim=1, keepdim=True)
    Wm = torch.stack([flat(priv(j), ks) for j in wp]); Wm = Wm / Wm.norm(dim=1, keepdim=True)
    off = ~torch.eye(len(new), dtype=bool)
    print(f"s{s_}: mean |B_perp| conv {sum(nc)/len(nc):.2f} vs WP {sum(nw)/len(nw):.2f}; pairwise cos among externals conv {(C@C.T)[off].mean():.3f} vs WP {(Wm@Wm.T)[off].mean():.3f}", flush=True)
