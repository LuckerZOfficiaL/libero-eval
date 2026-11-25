#!/usr/bin/env python3
"""Per-module multi-task merge operators for LoRA deltas D_i = B_i A_i (EXP-110).

Every function takes the group's factors Bs = [B_i (m x r)], As = [A_i (r x n)] for ONE module (fp32, on device)
plus hyperparameters, and returns the dense merged update dW (m x n). SVD-based methods work in core space (QR of
the stacked factors), so nothing wider than the stacked rank is decomposed, and use the NUMERICAL rank of LoRA sums
(the null-space bug: full-matrix SVDs return min(m, n) pairs of which only <= sum of ranks are real).
Specs follow the official code / papers (see EXPERIMENTS.md EXP-110 and memory lit-multitask-merging):
  avg, ta, ties (per-tensor top-k variant), dare_ties, tsv, iso_c, iso_cts, knots_ties, cart, ct_merge,
  ours (pool S16 + alpha * sum of two-sided residuals) and its ablations (pgc, group, lgo); cusd (EXP-146).
"""
import math
import torch

EPS_REL = 1e-6

def _stack(Bs, As):
    return torch.cat(Bs, 1), torch.cat(As, 0)

def _core_svd(B, A):
    """SVD of B @ A via QR of the factors; returns U (m x r), s (r), V (n x r) with numerical rank r."""
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    u, s, vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
    r = int((s > EPS_REL * s[0]).sum()) if s.numel() and s[0] > 0 else 0
    return Qb @ u[:, :r], s[:r], Qa @ vh[:r].T

def _polar(X):
    """orthonormalize columns (closest orthonormal matrix, as in TSV-M / Iso-CTS whitening)"""
    P, _, Qh = torch.linalg.svd(X, full_matrices=False)
    return P @ Qh

# ---------------------------------------------------------------------------------------------- simple baselines
def avg(Bs, As):
    B, A = _stack(Bs, As); return (B @ A) / len(Bs)

def ta(Bs, As, lam):
    B, A = _stack(Bs, As); return lam * (B @ A)

def _ties_stream(tau_fn, k, topk, lam):
    """TIES with per-tensor trimming, streamed over tasks (two passes, never k dense copies at once).
    tau_fn(i) -> dense task update i (deterministic). topk = fraction kept per task (e.g. 0.2)."""
    def trimmed(i):
        t = tau_fn(i)
        if topk < 1.0:
            a = t.abs().flatten(); kk = max(1, int(round(topk * a.numel())))
            thr = torch.topk(a, kk, sorted=False).values.min()
            t = torch.where(t.abs() >= thr, t, torch.zeros_like(t))
        return t
    S = None
    for i in range(k):
        t = trimmed(i); S = t.clone() if S is None else S.add_(t)
    sign = torch.sign(S); sign[sign == 0] = 1.0; del S
    num = torch.zeros_like(sign); cnt = torch.zeros_like(sign)
    for i in range(k):
        t = trimmed(i); agree = (torch.sign(t) == sign) & (t != 0)
        num.add_(torch.where(agree, t, torch.zeros_like(t))); cnt.add_(agree.float())
    return lam * num / cnt.clamp(min=1.0)

def ties(Bs, As, topk=0.2, lam=1.0):
    return _ties_stream(lambda i: Bs[i] @ As[i], len(Bs), topk, lam)

def dare_ties(Bs, As, p=0.5, topk=0.2, lam=1.0, seed=0, salt=0):
    def tau(i):
        t = Bs[i] @ As[i]
        g = torch.Generator(device=t.device).manual_seed(seed * 1000003 + salt * 101 + i)
        keep = (torch.rand(t.shape, generator=g, device=t.device) >= p).float()
        return t * keep / (1.0 - p)
    return _ties_stream(tau, len(Bs), topk, lam)

# ---------------------------------------------------------------------------------------------- SVD baselines
def tsv(Bs, As, alpha=1.0):
    m, n = Bs[0].shape[0], As[0].shape[1]; T = len(Bs)
    kt = min(16, int(min(m, n) / T))
    if kt == 0: return torch.zeros(m, n, device=Bs[0].device)
    Us, Ss, Vs = [], [], []
    for B, A in zip(Bs, As):
        U, s, V = _core_svd(B, A); q = min(kt, s.numel())
        if q: Us.append(U[:, :q]); Ss.append(s[:q]); Vs.append(V[:, :q])
    if not Us: return torch.zeros(m, n, device=Bs[0].device)
    U, S, V = torch.cat(Us, 1), torch.cat(Ss), torch.cat(Vs, 1)
    return alpha * (_polar(U) * S) @ _polar(V).T

def iso_c(Bs, As, alpha=1.0):
    B, A = _stack(Bs, As); U, s, V = _core_svd(B, A)
    if s.numel() == 0: return torch.zeros(B.shape[0], A.shape[1], device=B.device)
    return alpha * s.mean() * (U @ V.T)

def iso_cts(Bs, As, alpha=1.0, frac=0.8):
    B, A = _stack(Bs, As); U, s, V = _core_svd(B, A); T = len(Bs); r = s.numel()
    if r == 0: return torch.zeros(B.shape[0], A.shape[1], device=B.device)
    kc = max(1, int(frac * r)); st = (r - kc) // T
    Uc, Vc, sc = U[:, :kc], V[:, :kc], s[:kc]
    Ul, Vl, sl = [Uc], [Vc], [sc]
    if st > 0:
        for Bi, Ai in zip(Bs, As):
            Ui, si, Vi = _core_svd(Bi - Uc @ (Uc.T @ Bi), Ai)          # one-sided (left) residual, as in iso_cts
            q = min(st, si.numel())
            if q: Ul.append(Ui[:, :q]); Vl.append(Vi[:, :q]); sl.append(si[:q])
    Us, Vs, ss = torch.cat(Ul, 1), torch.cat(Vl, 1), torch.cat(sl)
    return alpha * ss.mean() * (_polar(Us) @ _polar(Vs).T)

def knots_ties(Bs, As, topk=0.2, lam=0.6):
    B = torch.cat(Bs, 1); Qb, Rb = torch.linalg.qr(B)                   # column-concat [D_1..D_k] = Qb [Rb_i A_i]
    r_ = Bs[0].shape[1]; blocks = [Rb[:, i * r_:(i + 1) * r_] for i in range(len(Bs))]
    G = sum(Bi @ (Ai @ Ai.T) @ Bi.T for Bi, Ai in zip(blocks, As))    # Gram of the concatenation in core space
    ev, W = torch.linalg.eigh(G); ev, W = ev.flip(0), W.flip(1)
    keep = ev.clamp(min=0).sqrt() > 1e-5
    W = W[:, keep]
    U = Qb @ W
    sV = [W.T @ (Bi @ Ai) for Bi, Ai in zip(blocks, As)]              # = diag(s) V_i^T, r x n
    merged = _ties_stream(lambda i: sV[i], len(sV), topk, lam)
    return U @ merged

def cart(Bs, As, lam=1.5, gamma=0.08):
    B, A = _stack(Bs, As); k = len(Bs); r_ = Bs[0].shape[1]
    Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
    cores = [Rb[:, i * r_:(i + 1) * r_] @ Ra[:, i * r_:(i + 1) * r_].T for i in range(k)]
    cbar = sum(cores) / k; out = cbar.clone()
    for c in cores:
        u, s, vh = torch.linalg.svd(c - cbar, full_matrices=False)
        rk = int((s > EPS_REL * s[0]).sum()) if s[0] > 0 else 0
        q = max(1, math.ceil(gamma * rk)) if rk else 0
        if q: out += lam * (u[:, :q] * s[:q]) @ vh[:q]
    return Qb @ out @ Qa.T

def ct_merge(Bs, As, gamma=0.6, kc=16, r=14):
    m, n = Bs[0].shape[0], As[0].shape[1]; T = len(Bs); dev = Bs[0].device
    kc = min(kc, min(m, n)); r = min(r, max(0, (min(m, n) - kc) // T))
    tasks = [_core_svd(B, A) for B, A in zip(Bs, As)]
    rr = min(16, min(t[1].numel() for t in tasks)) if r == 0 else r
    Ustack = torch.cat([t[0][:, :rr] for t in tasks], 1)               # projector average P = (1/T) sum U_t U_t^T
    ev, W = torch.linalg.eigh(Ustack.T @ Ustack / T); ev, W = ev.flip(0), W.flip(1)
    q = min(kc, int((ev > 1e-8).sum()))
    Uc = Ustack @ W[:, :q] / ev[:q].sqrt().clamp(min=1e-12) / math.sqrt(T)
    Uc = _polar(Uc)
    Dbar_B, Dbar_A = torch.cat(Bs, 1) / T, torch.cat(As, 0)
    Mc = (Uc.T @ Dbar_B) @ Dbar_A                                       # (1/T) sum U_c^T D_t, q x n
    Pc, Sc, Vch = torch.linalg.svd(Mc, full_matrices=False)
    Uc, Vc = Uc @ Pc, Vch.T
    Ul, Vl, rho = [Uc], [Vc], [torch.full((Sc.numel(),), float(math.sqrt((Sc ** 2).sum() / max(1, Sc.numel()))), device=dev)]
    if r > 0:
        for (Ut, st, Vt) in tasks:
            Rt_B = (Ut[:, :r] * st[:r]) - Uc @ (Uc.T @ (Ut[:, :r] * st[:r]))   # one-sided residual of the top-r part
            Ur, sr, Vr = _core_svd(Rt_B, Vt[:, :r].T)
            q2 = min(r, sr.numel())
            if q2:
                Ul.append(Ur[:, :q2]); Vl.append(Vr[:, :q2])
                rho.append(torch.full((q2,), float(math.sqrt((sr ** 2).sum() / r)), device=dev))
    Ut_, Vt_, D = _polar(torch.cat(Ul, 1)), _polar(torch.cat(Vl, 1)), torch.cat(rho)
    return gamma * (Ut_ * D) @ Vt_.T

# ---------------------------------------------------------------------------------------------- ours + ablations
def _resid_sum(Bs, As, Un, V):
    M_B, M_A = _stack(Bs, As)
    M = M_B @ M_A
    return M, M - Un @ ((Un.T @ M_B) @ (M_A @ V)) @ V.T

def ours(Bs, As, alpha, Usig, Vp):
    """pool S16 (= truncated SVD of the pool mean, U Sigma V^T) + alpha * sum of two-sided residuals"""
    Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
    _, R = _resid_sum(Bs, As, Un, Vp)
    return Usig @ Vp.T + alpha * R

def ours_pgc(Bs, As, alpha, Usig, Vp):
    """pool subspace, GROUP coefficients: P(mean_G D) + alpha * sum r_i (recovers the expert at k=1, alpha=1)"""
    Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
    M, R = _resid_sum(Bs, As, Un, Vp)
    return Un @ (Un.T @ M @ Vp) @ Vp.T / len(Bs) + alpha * R

def _top16(Bs, As, scale):
    U, s, V = _core_svd(torch.cat(Bs, 1) * scale, torch.cat(As, 0)); q = min(16, s.numel())
    return U[:, :q] * s[:q], V[:, :q]

def ours_group(Bs, As, alpha):
    """subspace + common term from the GROUP mean only (no pool information)"""
    Usig, Vp = _top16(Bs, As, 1.0 / len(Bs)); return ours(Bs, As, alpha, Usig, Vp)

def ours_lgo(Bs, As, alpha, pool_Bs, pool_As):
    """subspace + common term from the pool mean EXCLUDING the group (leave-group-out)"""
    Usig, Vp = _top16(pool_Bs, pool_As, 1.0 / len(pool_Bs)); return ours(Bs, As, alpha, Usig, Vp)

# ---------------------------------------------------------------------------------------------- candidate new methods (EXP-112)
def _residuals(Bs, As, Un, V):
    """dense two-sided residuals r_i = D_i - Un Un' D_i V V' (pool subspace)"""
    return [B @ A - Un @ ((Un.T @ B) @ (A @ V)) @ V.T for B, A in zip(Bs, As)]

def ours_ties(Bs, As, lam, Usig, Vp, topk=0.2):
    """pool S16 (exact, once) + lam * TIES(trim / sign-elect / disjoint-mean) over the RESIDUALS only.
    TIES on full deltas is dominated by the shared part; here the sign vote acts on the private parts."""
    Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
    R = _residuals(Bs, As, Un, Vp)
    return Usig @ Vp.T + _ties_stream(lambda i: R[i], len(R), topk, lam)

def ours_rm(Bs, As, gamma, Usig, Vp, gdiag):
    """pool S16 + gamma * input-weighted mean of residuals (diagonal RegMean on the residuals):
    column j of the merged residual = sum_i g_ij r_i[:, j] / sum_i g_ij, g_ij = task i's mean squared input feature j
    under its own policy on its demos. Uniform g gives the plain mean residual (== ours at alpha = 1/k when gamma = 1)."""
    Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
    R = _residuals(Bs, As, Un, Vp)
    G = torch.stack(gdiag)                                                   # [k, n]
    W = G / G.sum(0, keepdim=True).clamp(min=1e-30)
    return Usig @ Vp.T + gamma * sum(r * w[None, :] for r, w in zip(R, W))

# ---------------------------------------------------------------------------------------------- continual-merging baselines (EXP-116)
def maxabs(Bs, As, lam=1.0):
    """MaxAbs (training-free counterpart of MagMax, Marczak et al. ECCV 2024): per entry, keep the task update with the
    largest magnitude (its sign kept); lam * result.  (MagMax proper needs sequential fine-tuning, not applicable.)"""
    best = Bs[0] @ As[0]
    for B, A in zip(Bs[1:], As[1:]):
        t = B @ A; best = torch.where(t.abs() > best.abs(), t, best)
    return lam * best

def opcm_unnorm(Bs, As, alpha=0.5):
    """OPCM (Tang et al. 2025; fusion_bench opcm.py), sequential in list order, UNNORMALISED accumulator U_T:
    U_1 = tau_1; for t >= 2: SVD of U_{t-1} = u s v^T; r = first index with cumsum(s)/sum(s) > alpha;
    cleaned = tau_t minus its components on the top-left r x r block and on the diagonal pairs u_i v_i^T (within the range
    of U_{t-1}; null-space diagonal terms of the full SVD are basis-arbitrary and ignored); U_t = U_{t-1} + cleaned.
    The caller rescales globally: merged = U_T * mean_i ||tau_i|| / ||U_T|| (norm-based lambda_t of the paper)."""
    L, R = Bs[0], As[0]                                           # U = L @ R (factored, recompressed each step)
    for B, A in zip(Bs[1:], As[1:]):
        Ql, Rl = torch.linalg.qr(L); Qr, Rr = torch.linalg.qr(R.T)
        u_, s, vh = torch.linalg.svd(Rl @ Rr.T, full_matrices=False)
        q = int((s > 1e-7 * s[0]).sum()) if s.numel() and s[0] > 0 else 0
        u, s, v = Ql @ u_[:, :q], s[:q], Qr @ vh[:q].T
        r = int((s.cumsum(0) / s.sum() > alpha).float().argmax()) if q else 0
        UB, AV = u.T @ B, A @ v                                    # (q x 16), (16 x q)
        P = UB @ AV                                                # u^T tau v, q x q
        keep = torch.ones_like(P); keep[:r, :r] = 0; keep.fill_diagonal_(0)
        removed = P * (1 - keep)                                   # block + diagonal components
        L = torch.cat([L, B, -(u @ removed)], 1); R = torch.cat([R, A, v.T], 0)
        Ql, Rl = torch.linalg.qr(L); Qr, Rr = torch.linalg.qr(R.T)  # recompress to numerical rank
        u_, s2, vh = torch.linalg.svd(Rl @ Rr.T, full_matrices=False)
        q2 = int((s2 > 1e-7 * s2[0]).sum()); L = (Ql @ u_[:, :q2]) * s2[:q2]; R = (Qr @ vh[:q2].T).T
    return L @ R

# ---------------------------------------------------------------------------------------------- WUDI (EXP-130)
def wudi(Bs, As, iters=300, lr=1e-5, init="sum", return_curve=False):
    """WUDI-Merging (Cheng et al., ICML 2025, arXiv 2503.08099), data-free: per linear layer minimise
    sum_i ||(tau_m - tau_i) tau_i^T||_F^2 / ||tau_i||_F^2 with Adam (lr 1e-5, 300 iterations), tau_m initialised at the
    SUM of the task vectors, no rescaling (official defaults). LoRA: tau_i = B_i A_i (restored, as in the paper).
    Factored and vectorised over tasks: (tau_m - tau_i) tau_i^T = Y_i B_i^T, Y_i = tau_m A_i^T - B_i (A_i A_i^T)."""
    Bs = [b.float() for b in Bs]; As = [a.float() for a in As]; k, r = len(Bs), As[0].shape[0]
    Aall = torch.cat(As, 0)                                                     # (k r) x in
    G = torch.stack([b.T @ b for b in Bs]); H = torch.stack([a @ a.T for a in As])
    w = 1.0 / ((G * H.transpose(1, 2)).sum((1, 2)).abs() + 1e-30)               # 1 / ||B_i A_i||_F^2
    BH = torch.cat([b @ h for b, h in zip(Bs, H)], 1)                           # out x (k r) = [tau_i A_i^T]
    tau = sum(b @ a for b, a in zip(Bs, As))
    if init != "sum": tau = tau / k
    with torch.enable_grad():                                                   # callers (Builder.build) run under no_grad
        tau = tau.detach().clone().requires_grad_(True); opt = torch.optim.Adam([tau], lr=lr); curve = []
        for it in range(iters):
            Y = (tau @ Aall.T - BH).view(tau.shape[0], k, r)                     # out x k x r
            loss = (torch.einsum("oir,irs->ois", Y, G) * Y).sum((0, 2)) @ w
            opt.zero_grad(); loss.backward(); opt.step()
            if return_curve and (it % 50 == 0 or it == iters - 1): curve.append(float(loss))
    return (tau.detach(), curve) if return_curve else tau.detach()

# ---------------------------------------------------------------------------------------------- CUSD (EXP-146)
def _kde_valley(d):
    """official find_breakpoint_kde: Gaussian KDE (Scott) of the dispersions > 1e-8 on a 1000-point grid, local minima
    (argrelmin, order 20), the one nearest the mid-range; None when there are < 10 points, the KDE fails or no valley
    exists (official classify_subspaces then keeps every direction common)."""
    import numpy as np
    from scipy.stats import gaussian_kde
    from scipy.signal import argrelmin
    v = d[d > 1e-8]
    if len(v) < 10: return None
    try: kde = gaussian_kde(v, bw_method="scott")
    except Exception: return None
    x = np.linspace(v.min(), v.max(), 1000); mi = argrelmin(kde(x), order=20)[0]
    return float(x[mi[np.argmin(np.abs(x[mi] - (v.min() + v.max()) / 2))]]) if len(mi) else None

def cusd(Bs, As, beta, F="avg", mode="rel", lam=None, topk=0.2, pi=1e-3, stats=None):
    """CUSD (Wang, Zou, Li, NeurIPS 2026, arXiv 2611.04108; official code github.com/stwang666/CUSD, followed step by step):
    shared RIGHT basis V = eigenvectors (descending) of the symmetrised pairwise Gram-quotient average of the regularised
    Grams G_i = D_i^T D_i + pi I (HO-GSVD-derived); energies s_ik = ||D_i v_k||; dispersion d_k = std_i s_ik ('abs') or
    std_i s_ik / mean_i s_ik ('rel'); KDE valley -> common (d_k < valley) / unique, no valley -> all common.
    Merge = F(C_1..C_K) + beta * sum_i R_i, C_i = D_i V_C V_C^T, R_i = D_i - C_i; F = avg | ta (lam) | ties (topk, lam).
    LoRA: solved in the joint row space of the A_i (dim <= sum of ranks) instead of R^n. Outside it every G_i = pi I, so
    the quotient matrix is block diagonal (identity there) and those directions carry no energy: the official code drops
    them from the KDE (d <= 1e-8) and keeps them common with nothing in them -- same result, no n x n solves (n <= 16384).
    stats: optional list, gets one dict per call (directions, unique count, unique share of the energy; per task: its energy
    and the share of it in the unique directions)."""
    import numpy as np
    k = len(Bs); M_B, M_A = _stack(Bs, As)
    _, s, Vh = torch.linalg.svd(M_A, full_matrices=False)
    Q = Vh[: int((s > EPS_REL * s[0]).sum())].T                                  # n x R joint row space
    X = [(B @ (A @ Q)).double() for B, A in zip(Bs, As)]                         # D_i Q
    I = torch.eye(Q.shape[1], dtype=torch.float64, device=Q.device)
    G = torch.stack([x.T @ x + pi * I for x in X])
    # official: sum over pairs i < j of G_j^-1 G_i + G_i^-1 G_j = sum over all i != j of G_j^-1 G_i = (sum_j G_j^-1)(sum_i G_i) - k I
    S = (torch.cholesky_inverse(torch.linalg.cholesky(G)).sum(0) @ G.sum(0) - k * I) / (k * (k - 1))
    _, W = torch.linalg.eigh((S + S.T) / 2); W = W.flip(1)
    sig = torch.stack([torch.linalg.norm(x @ W, dim=0) for x in X]).cpu().numpy()   # K x R
    mu = sig.mean(0); d = (sig / np.where(mu > 1e-12, mu, 1e-12)).std(0) if mode == "rel" else sig.std(0)
    th = _kde_valley(d); com = np.ones(len(d), bool) if th is None else d < th
    if stats is not None:                                                        # energies = squared norms (sum over directions = ||D_i||_F^2)
        e2 = sig ** 2; ti = e2.sum(1)
        stats.append({"R": len(d), "n_unique": int((~com).sum()), "unique_energy": float(e2[:, ~com].sum() / max(ti.sum(), 1e-30)),
                      "task_energy": ti.tolist(), "task_unique": (e2[:, ~com].sum(1) / np.maximum(ti, 1e-30)).tolist()})
    Vc = Q @ W[:, torch.from_numpy(com).to(W.device)].to(Q.dtype)                # n x |C|
    Cs = M_B @ (M_A @ Vc) @ Vc.T                                                 # sum_i C_i
    if F == "avg": Fc = Cs / k
    elif F == "ta": Fc = lam * Cs
    elif F == "ties": Fc = _ties_stream(lambda i: Bs[i] @ (As[i] @ Vc) @ Vc.T, k, topk, lam)
    else: raise ValueError(F)
    return Fc + beta * (M_B @ M_A - Cs)
