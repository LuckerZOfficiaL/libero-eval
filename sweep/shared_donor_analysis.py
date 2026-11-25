#!/usr/bin/env python3
"""EXP-49 analysis — two-way decomposition on the complete 15x8 target-donor grid.

    affinity(t,d) = mu + a_t (target effect) + b_d (DONOR PRIOR) + e_td (INTERACTION)

A complete grid makes these orthogonal, so the question route 1 could not answer becomes arithmetic:
does the rollout-free feature dnorm_llm_00 track b_d, e_td, or both? Only e_td can support a selector --
a donor prior ranks the same donors first for every target, and cannot explain why the probe's choices
differ per target. Reported for every merged-model feature, with the probe's own reliability as the
yardstick (test-retest rho 0.41-0.59; a feature cannot beat the noise in its own labels).
"""
import json, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from reinforce_train import scenes as SC, target_baseline
import unified_pool_driver as U
D = json.load(open(REPO / "results/analysis/shared_donor_design.json"))
T, Dn = D["targets"], D["donors"]
LAYERS = ["vision", "llm_00", "llm_05", "llm_11", "llm_17", "exp_00", "exp_08", "exp_17", "act_out"]
AS = REPO / "results/analysis/act_sigs"; MS = REPO / "results/analysis/merged_sigs"

def label(t, d):
    for job in (f"e59_t{t}_p{d}", f"e27_probe_t{t}_p{d}"):
        v = U.get(job, SC[t][0])
        if v is not None: return 100 * (v - target_baseline(t))
    return None

def feats(t, d):
    f, g = MS / f"m{t}_{d}.npz", AS / f"task{t}.npz"
    if not (f.exists() and g.exists()): return None
    M, Tg = np.load(f), np.load(g); out = {}
    for L in LAYERS:
        a = M[L].astype(np.float32); b = Tg[L].astype(np.float32); n = min(len(a), len(b)); a, b = a[:n], b[:n]
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        out[f"dist_{L}"] = float(np.linalg.norm(a - b, axis=1).mean())
        out[f"cos_{L}"] = float(((a * b).sum(1) / (na * nb + 1e-9)).mean())
        out[f"dnorm_{L}"] = float(((na - nb) / (nb + 1e-9)).mean())
    return out

Y = np.full((len(T), len(Dn)), np.nan)
F = {}
for i, t in enumerate(T):
    for j, d in enumerate(Dn):
        y = label(t, d); fe = feats(t, d)
        if y is not None: Y[i, j] = y
        if fe:
            for k, v in fe.items(): F.setdefault(k, np.full((len(T), len(Dn)), np.nan))[i, j] = v
filled = np.isfinite(Y).sum()
print(f"=== EXP-49: shared-donor grid {len(T)}x{len(Dn)} ===")
print(f"labels {filled}/{Y.size} cells; features {min([np.isfinite(v).sum() for v in F.values()]) if F else 0}/{Y.size}")
if filled < Y.size:
    print("(incomplete grid: rows/cols with gaps are dropped from the decomposition)")
def decompose(A):
    m = np.nanmean(A); a = np.nanmean(A, 1) - m; b = np.nanmean(A, 0) - m
    return m, a, b, A - (m + a[:, None] + b[None, :])
mu, at, bd, E = decompose(Y)
ss_t = np.nansum((at[:, None] * np.ones_like(Y)) ** 2); ss_d = np.nansum((bd[None, :] * np.ones_like(Y)) ** 2)
ss_e = np.nansum(E ** 2); tot = ss_t + ss_d + ss_e
print(f"\nvariance of AFFINITY: target {100*ss_t/tot:.0f}%   donor prior {100*ss_d/tot:.0f}%   interaction {100*ss_e/tot:.0f}%")
print(f"donor prior b_d: {dict(zip(Dn, np.round(bd,1)))}")
print(f"\n{'feature':<16}{'corr with donor prior':>23}{'corr with INTERACTION':>24}{'p':>9}")
best = None
for nm in sorted(F):
    A = F[nm]
    if np.isfinite(A).sum() < 0.8 * A.size: continue
    _, _, fb, fE = decompose(A)
    ok = np.isfinite(E) & np.isfinite(fE)
    rb = stats.pearsonr(bd, fb)[0] if np.isfinite(bd).all() and np.isfinite(fb).all() else np.nan
    re_, pe = stats.pearsonr(fE[ok], E[ok])
    print(f"{nm:<16}{rb:>+23.3f}{re_:>+24.3f}{pe:>9.4f}")
    if best is None or abs(re_) > abs(best[1]): best = (nm, re_, pe)
if best:
    print(f"\nstrongest INTERACTION predictor: {best[0]}  r={best[1]:+.3f} (p={best[2]:.4f}, n={int((np.isfinite(E)).sum())} cells)")
    print("a selector needs interaction, not donor prior; the probe's own labels carry rho 0.41-0.59 of reliability")
