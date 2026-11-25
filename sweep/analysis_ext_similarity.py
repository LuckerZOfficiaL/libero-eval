# weight-space cosine between full deltas D = b @ a of skills, energy-weighted over layers (per layer <Di,Dj>_F via r x r traces)
import sys, pathlib, numpy as np, torch, itertools
REPO = pathlib.Path("/home/lzhou/libero-eval"); sys.path[:0] = [str(REPO / "sweep"), str(REPO / "scripts")]
import exp116_continual as C, exp112_bench as E, unified_pool_driver as U
torch.set_num_threads(16)
ext = sorted({e for s in range(4) for e in C.sequence(s)[1]}); lib = sorted(C.sequence(0)[0])[:8]
ids = ext + lib
def fac(i):
    e = E.LIB[i]; return U.fold(e["key"]) if e["ext"] else E.load(e["key"])
F = {i: {s_: (f[0].float().cuda(), f[1].float().cuda()) for s_, f in fac(i).items()} for i in ids}
stems = sorted(set.intersection(*[set(F[i]) for i in ids]))
n = len(ids); G = np.zeros((n, n)); VLM = np.zeros((n, n))
for s_ in stems:
    for x, y in itertools.combinations_with_replacement(range(n), 2):
        bi, ai = F[ids[x]][s_]; bj, aj = F[ids[y]][s_]
        if bi.shape != bj.shape or ai.shape[1] != aj.shape[1]:
            if bi.shape[0] == aj.shape[1]: bj, aj = aj.T, bj.T            # other orientation
            else: continue
        v = float(torch.trace((bi.T @ bj) @ (aj @ ai.T)))
        G[x, y] = G[y, x] = G[x, y] + v
        if "gemma_expert" not in s_: VLM[x, y] = VLM[y, x] = VLM[x, y] + v
def cos(M):
    d = np.sqrt(np.clip(np.diag(M), 1e-30, None)); return M / d[:, None] / d[None, :]
for name, M in (("all layers", G), ("VLM layers", VLM)):
    c = cos(M); su = [E.LIB[i]["suite"] if E.LIB[i]["ext"] else "lib90" for i in ids]
    def m(f): v = [c[x, y] for x in range(n) for y in range(x + 1, n) if f(x, y)]; return np.mean(v), len(v)
    print(f"{name}: cos same-suite externals {m(lambda x,y: su[x]==su[y]!='lib90')[0]:+.3f}  cross-suite externals {m(lambda x,y: su[x]!=su[y] and 'lib90' not in (su[x],su[y]))[0]:+.3f}"
          f"  library-library {m(lambda x,y: su[x]==su[y]=='lib90')[0]:+.3f}  external-library {m(lambda x,y: (su[x]=='lib90')!=(su[y]=='lib90'))[0]:+.3f}")
    for s in sorted(set(su)):
        if s != "lib90": print(f"   {s}: {m(lambda x,y: su[x]==su[y]==s)[0]:+.3f} (pairs {m(lambda x,y: su[x]==su[y]==s)[1]})")
print("externals:", len(ext), "layers:", len(stems))
