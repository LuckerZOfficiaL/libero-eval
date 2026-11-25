#!/usr/bin/env python3
"""EXP-88/89/90 stage 1 (weights only, no rollouts): the linear algebra of the pool, per target.

One pass over the 458 adapted modules of all 76 fleet adapters accumulates everything the three questions
need, energy-weighted across modules (numerators and denominators summed separately, as in the S16 test):

  EXP-88  more than one shared component?
    (a) stacked residual subspace, 4-fold held-out: top-k left/right singular directions of the stacked
        residuals of 57 tasks; how much of a HELD-OUT task's residual lies in them (two-sided), vs the same
        for held-in tasks, vs chance k^2/(d_out d_in). held-out ~ held-in >> chance  => shared directions.
    (b) cluster-level: the same with the subspace built from the task's semantic cluster (residual-cosine
        average linkage, k=5 and k=8, task and its duplicate excluded) vs a random group of equal size.
  EXP-89  does a target share more with some tasks than others?
    (a) residual subspace of its n nearest residual neighbours (n = 4, 8, 16) vs n random tasks
    (b) soup of its n nearest neighbours' FULL deltas, top-16/32 two-sided directions: capture of the
        target's own full delta, vs n random tasks, vs the pool S16 (0.80)
  EXP-90  per-task rank of the shared component
    per task: core C_t = U16' D_t V16 (16x16) -> capture curve k=1..16, per-direction loadings, and the
    76x76 similarity of cores (is each task's own shared part the same, or closer to some tasks?)

Also the 76x76 full-delta and residual inner products (cosines, cluster-mean-vector tests).
Duplicate adapters (t7=t18, t29=t62, identical md5) are never used as each other's neighbours/group-mates.
Output: results/analysis/linalg_structure.npz (+ .json meta). Analysis: sweep/linalg_report.py.
"""
import json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import safetensors.torch as st
import apply_lora as al
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

N = 76
DUP = {7: 18, 18: 7, 29: 62, 62: 29}
KR = [4, 8, 16, 32]            # residual-subspace ranks
KS = [16, 32]                  # soup ranks
NN = [4, 8, 16]                # neighbourhood sizes
NRAND = 3                      # random groups per (target, size)
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0      # modules, for smoke tests
SHARD = sys.argv[3] if len(sys.argv) > 3 else "0/1"        # "i/S": this process handles modules with index % S == i
DT = torch.float64

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

# ---------------- groups (fixed before looking at any capture) ----------------
C = np.load(REPO / "results/analysis/residual_cosines.npz")["C"]
def excl(s): return {s, DUP.get(s, -1)}
rng = np.random.default_rng(88)
nn = {n: [] for n in NN}; rnd = {n: [] for n in NN}
for s in range(N):
    order = [g for g in np.argsort(-C[s]) if g not in excl(s)]
    others = [g for g in range(N) if g not in excl(s)]
    for n in NN:
        nn[n].append(order[:n])
        rnd[n].append([sorted(rng.choice(others, n, replace=False).tolist()) for _ in range(NRAND)])
D = np.clip(1 - C.copy(), 0, None); np.fill_diagonal(D, 0); D = (D + D.T) / 2
L = linkage(squareform(D, checks=False), method="average")
clus = {kc: fcluster(L, t=kc, criterion="maxclust") for kc in (5, 8)}
cgrp = {kc: [] for kc in clus}; crnd = {kc: [] for kc in clus}
for kc, lab in clus.items():
    for s in range(N):
        g = [x for x in range(N) if lab[x] == lab[s] and x not in excl(s)]
        cgrp[kc].append(g)
        others = [x for x in range(N) if x not in excl(s)]
        crnd[kc].append([sorted(rng.choice(others, len(g), replace=False).tolist()) if len(g) else [] for _ in range(NRAND)])
fold = np.zeros(N, int)
perm = rng.permutation([t for t in range(N) if t not in (18, 62)])
for i, t in enumerate(perm): fold[t] = i % 4
fold[18], fold[62] = fold[7], fold[29]

# ---------------- accumulators ----------------
A = {k: np.zeros(v) for k, v in dict(
    dn2=N, rn2=N, core_curve=(N, 16), core_row=(N, 16), core_col=(N, 16),
    coreG=(N, N), coreCG=(N, N), ipR=(N, N), ipD=(N, N),
    st_ho=(N, len(KR)), st_hi=(N, len(KR)), st_ho_left=(N, len(KR)), chance=(N, len(KR)),
    **{f"r_{nm}": (N, len(KR)) for nm in
       ["nn4", "nn8", "nn16", "rnd4", "rnd8", "rnd16", "clu5", "clu8", "rclu5", "rclu8"]},
    **{f"s_{nm}": (N, len(KS)) for nm in
       ["nn4", "nn8", "nn16", "rnd4", "rnd8", "rnd16", "clu5", "clu8", "rclu5", "rclu8"]},
    stack_spec=64, rn2_g=N, dn2_g=N, core16_g=N).items()}
GROUP_EVERY = 4        # group analyses (nn / random / cluster) on a stratified quarter of the modules
A["stack_tot"] = np.zeros(1)

def topk_eigh(G, k, rel=1e-9):
    """G [..., m, m] PSD -> top-k eigvecs, eigvals (descending) in fp64, and 1/sqrt(eigval) with directions
    below rel*max masked to 0 (near-null directions are numerically meaningless and capture nothing)."""
    w, V = torch.linalg.eigh(G)
    w = w.flip(-1)[..., :k]; V = V.flip(-1)[..., :k]
    ok = w > rel * w[..., :1].clamp(min=1e-300)
    inv = torch.where(ok, w.clamp(min=1e-300).rsqrt(), torch.zeros_like(w))
    return V, w, inv

def resid_capture(GX, GY, Z, idx_g, s_list, k_list):
    """Two-sided capture numerators of residual s by the stacked-residual subspace of group idx_g:
    U_k = X_G W diag(l^-1/2), V_k = Y_G W' diag(l'^-1/2) (orthonormal); U_k' r_s V_k = diag . W'(X_G'P_s)(Y_s'Y_G)W' . diag
    idx_g: [B, m] task indices (same size across batch); s_list: [B]. Returns [B, len(k_list)] and left-only."""
    B_, m = idx_g.shape
    cols = (idx_g[:, :, None] * 32 + torch.arange(32, device=dev)).reshape(B_, m * 32)
    scol = (s_list[:, None] * 32 + torch.arange(32, device=dev))
    kmax = min(max(k_list), m * 32)
    if B_ > 1 and bool((idx_g == idx_g[0]).all()):        # one group for all s: diagonalise ONCE
        c0 = cols[:1]
        Wl, _, il = topk_eigh(GX[c0[:, :, None], c0[:, None, :]], kmax); Wr, _, ir = topk_eigh(GY[c0[:, :, None], c0[:, None, :]], kmax)
        Wl, il, Wr, ir = Wl.expand(B_, -1, -1), il.expand(B_, -1), Wr.expand(B_, -1, -1), ir.expand(B_, -1)
    else:
        Wl, _, il = topk_eigh(GX[cols[:, :, None], cols[:, None, :]], kmax); Wr, _, ir = topk_eigh(GY[cols[:, :, None], cols[:, None, :]], kmax)
    Zgs = Z[cols[:, :, None], scol[:, None, :]]                  # X_G' P_s   [B, 32m, 32]
    Gys = GY[scol[:, :, None], cols[:, None, :]]                 # Y_s' Y_G   [B, 32, 32m]
    Xgs = GX[cols[:, :, None], scol[:, None, :]]                 # X_G' X_s
    L_ = (Wl.transpose(1, 2) @ Zgs) * il[:, :, None]             # [B, k, 32]
    R_ = (Gys @ Wr) * ir[:, None, :]                             # [B, 32, k]
    M = L_ @ R_                                                  # [B, k, k] = U_k' r_s V_k
    Lo = ((Wl.transpose(1, 2) @ Xgs) * il[:, :, None]).pow(2).sum(-1)   # [B, k] left-only per direction
    out, outl = [], []
    for k in k_list:
        kk = min(k, kmax)
        out.append(M[:, :kk, :kk].pow(2).sum((1, 2))); outl.append(Lo[:, :kk].sum(1))
    return torch.stack(out, 1), torch.stack(outl, 1)

def soup_capture(GB, GA, idx_g, s_list, k_list):
    """capture numerators of the FULL delta of s by the top-k two-sided singular directions of the soup of
    group g (M = B_G A_G), built from orthonormal bases of B_G's column space and A_G's row space via eigh of
    their Grams (rank-deficient groups -- e.g. modules with d_in = 32 -- handled by masking null directions)."""
    B_, m = idx_g.shape
    cols = (idx_g[:, :, None] * 16 + torch.arange(16, device=dev)).reshape(B_, m * 16)
    scol = (s_list[:, None] * 16 + torch.arange(16, device=dev))
    Gb = GB[cols[:, :, None], cols[:, None, :]]; Ga = GA[cols[:, :, None], cols[:, None, :]]
    Wb, lb, ib = topk_eigh(Gb, m * 16); Wa, la, ia = topk_eigh(Ga, m * 16)
    sb = torch.where(ib > 0, lb.clamp(min=0).sqrt(), torch.zeros_like(lb)); sa = torch.where(ia > 0, la.clamp(min=0).sqrt(), torch.zeros_like(la))
    small = sb[:, :, None] * (Wb.transpose(1, 2) @ Wa) * sa[:, None, :]     # B_G A_G in the (Qb, Qa) bases
    Us, sv, Vsh = torch.linalg.svd(small)
    Gbs = GB[cols[:, :, None], scol[:, None, :]]                  # B_G' b_s   [B, 16m, 16]
    Gas = GA[scol[:, :, None], cols[:, None, :]]                  # a_s A_G'   [B, 16, 16m]
    left = (Wb.transpose(1, 2) @ Gbs) * ib[:, :, None]            # Qb' b_s
    right = (Gas @ Wa) * ia[:, None, :]                           # a_s Qa
    out = []
    for k in k_list:
        kk = min(k, m * 16)
        M = Us[:, :, :kk].transpose(1, 2) @ left @ right @ Vsh[:, :kk, :].transpose(1, 2)
        out.append(M.pow(2).sum((1, 2)))
    return torch.stack(out, 1)

def main():
    t0 = time.time()
    print(f"[la] loading {N} adapters ...", flush=True)
    AB = [load(t) for t in range(N)]
    stems = sorted(set.intersection(*[set(x) for x in AB]))
    if LIMIT: stems = stems[:: max(1, len(stems) // LIMIT)][:LIMIT]
    print(f"[la] {len(stems)} modules, loaded in {time.time()-t0:.0f}s", flush=True)
    ar = torch.arange(N, device=dev)
    si, sS = map(int, SHARD.split("/"))
    for mi, stem in enumerate(stems):
        if mi % sS != si: continue
        b = torch.stack([AB[t][stem][0] for t in range(N)]).to(dev, DT)     # [N, dout, 16]
        a = torch.stack([AB[t][stem][1] for t in range(N)]).to(dev, DT)     # [N, 16, din]
        dout, din = b.shape[1], a.shape[2]
        Ball = b.permute(1, 0, 2).reshape(dout, N * 16); Aall = a.reshape(N * 16, din)
        GB = Ball.T @ Ball; GA = Aall @ Aall.T
        # pool S16 (full-76 mean): explicit U16, V16
        Qb, Rb = torch.linalg.qr(Ball); Qa, Ra = torch.linalg.qr(Aall.T)
        Us, sv, Vh = torch.linalg.svd(Rb @ Ra.T)
        U16 = Qb @ Us[:, :16]; V16 = Qa @ Vh[:16].T
        # cores and full-delta energies
        Ub = U16.T @ b; aV = a @ V16; Cc = Ub @ aV                            # [N,16,16]
        dn2 = ((b.transpose(1, 2) @ b) * (a @ a.transpose(1, 2))).sum((1, 2))
        A["dn2"] += dn2.cpu().numpy()
        cum = torch.stack([Cc[:, :k, :k].pow(2).sum((1, 2)) for k in range(1, 17)], 1)
        A["core_curve"] += cum.cpu().numpy()
        A["core_row"] += Cc.pow(2).sum(2).cpu().numpy(); A["core_col"] += Cc.pow(2).sum(1).cpu().numpy()
        Cf = Cc.reshape(N, -1); A["coreG"] += (Cf @ Cf.T).cpu().numpy()
        Cm = Cf - Cf.mean(0, keepdim=True); A["coreCG"] += (Cm @ Cm.T).cpu().numpy()
        # full-delta inner products
        ipD = (GB.reshape(N, 16, N, 16).permute(0, 2, 1, 3) * GA.reshape(N, 16, N, 16).permute(0, 2, 1, 3)).sum((2, 3))
        A["ipD"] += ipD.cpu().numpy()
        # residual factors (operator form): r = [b, -U U'b] [a ; (aV)V']
        Bp = torch.cat([b, -(U16 @ Ub)], 2); Ap = torch.cat([a, aV @ V16.T], 1)       # [N,dout,32], [N,32,din]
        Qa_, Ra_ = torch.linalg.qr(Ap.transpose(1, 2)); X = Bp @ Ra_.transpose(1, 2)   # r = X Qa'
        Pb, Sb = torch.linalg.qr(Bp); Y = Ap.transpose(1, 2) @ Sb.transpose(1, 2)      # r = Pb Y'
        Xall = X.permute(1, 0, 2).reshape(dout, N * 32); Yall = Y.permute(1, 0, 2).reshape(din, N * 32)
        Pall = Pb.permute(1, 0, 2).reshape(dout, N * 32)
        GX = Xall.T @ Xall; GY = Yall.T @ Yall; Z = Xall.T @ Pall
        rn2 = GX.diagonal().reshape(N, 32).sum(1)
        A["rn2"] += rn2.cpu().numpy()
        Bpall = Bp.permute(1, 0, 2).reshape(dout, N * 32); Apall = Ap.reshape(N * 32, din)
        ipR = ((Bpall.T @ Bpall).reshape(N, 32, N, 32).permute(0, 2, 1, 3) *
               (Apall @ Apall.T).reshape(N, 32, N, 32).permute(0, 2, 1, 3)).sum((2, 3))
        A["ipR"] += ipR.cpu().numpy()
        for ki, k in enumerate(KR):
            A["chance"][:, ki] += (rn2 * min(k * k / (dout * din), 1.0)).cpu().numpy()
        # full-76 stacked spectrum (left side): how concentrated is residual energy in shared directions
        ev = torch.linalg.eigvalsh(GX).flip(0)[:64]
        A["stack_spec"] += ev.cpu().numpy(); A["stack_tot"] += float(GX.trace())
        # EXP-88a: 4-fold stacked residual subspace, held-out and held-in capture
        for f in range(4):
            tr = torch.tensor([t for t in range(N) if fold[t] != f], device=dev)
            te = torch.tensor([t for t in range(N) if fold[t] == f], device=dev)
            for part, key in ((te, "st_ho"), (tr, "st_hi")):
                g = tr[None, :].expand(len(part), -1)
                num, numl = resid_capture(GX, GY, Z, g, part, KR)
                # each task is held IN for 3 of the 4 folds -> average, not sum
                A[key][part.cpu().numpy()] += num.cpu().numpy() / (3.0 if key == "st_hi" else 1.0)
                if key == "st_ho": A["st_ho_left"][part.cpu().numpy()] += numl.cpu().numpy()
        # EXP-89 / 88b: neighbourhoods, random groups, clusters (residual subspace + full-delta soup)
        # on every GROUP_EVERY-th module; denominators accumulated on the same subset
        if mi % GROUP_EVERY: 
            if mi % 20 == 0: print(f"[la] {mi+1}/{len(stems)} modules ({dout}x{din})  {(time.time()-t0)/60:.1f} min", flush=True)
            continue
        A["rn2_g"] += rn2.cpu().numpy(); A["dn2_g"] += dn2.cpu().numpy(); A["core16_g"] += cum[:, 15].cpu().numpy()
        sidx = ar
        for n in NN:
            g = torch.tensor(nn[n], device=dev)
            A[f"r_nn{n}"] += resid_capture(GX, GY, Z, g, sidx, KR)[0].cpu().numpy()
            A[f"s_nn{n}"] += soup_capture(GB, GA, g, sidx, KS).cpu().numpy()
            for d in range(NRAND):
                g = torch.tensor([rnd[n][s][d] for s in range(N)], device=dev)
                A[f"r_rnd{n}"] += resid_capture(GX, GY, Z, g, sidx, KR)[0].cpu().numpy() / NRAND
                A[f"s_rnd{n}"] += soup_capture(GB, GA, g, sidx, KS).cpu().numpy() / NRAND
        for kc in clus:
            sizes = sorted({len(cgrp[kc][s]) for s in range(N)})
            for m in sizes:
                ss = [s for s in range(N) if len(cgrp[kc][s]) == m]
                if m < 2: continue
                sl = torch.tensor(ss, device=dev)
                g = torch.tensor([cgrp[kc][s] for s in ss], device=dev)
                A[f"r_clu{kc}"][ss] += resid_capture(GX, GY, Z, g, sl, KR)[0].cpu().numpy()
                A[f"s_clu{kc}"][ss] += soup_capture(GB, GA, g, sl, KS).cpu().numpy()
                for d in range(NRAND):
                    g = torch.tensor([crnd[kc][s][d] for s in ss], device=dev)
                    A[f"r_rclu{kc}"][ss] += resid_capture(GX, GY, Z, g, sl, KR)[0].cpu().numpy() / NRAND
                    A[f"s_rclu{kc}"][ss] += soup_capture(GB, GA, g, sl, KS).cpu().numpy() / NRAND
        if mi % 20 == 0 or mi == len(stems) - 1:
            el = time.time() - t0
            print(f"[la] {mi+1}/{len(stems)} modules ({dout}x{din})  {el/60:.1f} min", flush=True)
            np.savez(REPO / f"results/analysis/linalg_structure{'_smoke' if LIMIT else ''}{'' if sS == 1 else f'_s{si}'}.npz",
                     **A, done=mi + 1, KR=KR, KS=KS, NN=NN)
    np.savez(REPO / f"results/analysis/linalg_structure{'_smoke' if LIMIT else ''}{'' if sS == 1 else f'_s{si}'}.npz",
             **A, done=len(stems), KR=KR, KS=KS, NN=NN)
    meta = dict(nn={str(n): [[int(x) for x in g] for g in nn[n]] for n in NN}, clusters={str(kc): [int(x) for x in clus[kc]] for kc in clus},
                fold=[int(x) for x in fold], dup={str(k): v for k, v in DUP.items()}, modules=len(stems))
    json.dump(meta, open(REPO / f"results/analysis/linalg_structure{'_smoke' if LIMIT else ''}.json", "w"))
    print(f"[la] DONE {len(stems)} modules in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
