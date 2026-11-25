#!/usr/bin/env python3
"""Report for sweep/linalg_structure.py (EXP-88/89/90 stage 1). Usage: linalg_report.py [npz]"""
import json, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
f = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/analysis/linalg_structure.npz"
z = np.load(f); meta = json.load(open(str(f).replace(".npz", ".json"))) if pathlib.Path(str(f).replace(".npz", ".json")).exists() else {}
KR, KS, NN = list(z["KR"]), list(z["KS"]), list(z["NN"])
N = 76; DUP = {7: 18, 18: 7, 29: 62, 62: 29}
rn2, dn2 = z["rn2"], z["dn2"]
rg, dg_ = z["rn2_g"], z["dn2_g"]      # denominators on the module subset used for group analyses
try:
    from reinforce_train import target_baseline
    base = np.array([target_baseline(t) for t in range(N)])
except Exception:
    base = None
weak = np.where(base < 0.9)[0] if base is not None else np.arange(0)
keep = np.array([t for t in range(N) if t not in (18, 62)])      # duplicates counted once
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))
def m(x): return f"{np.mean(x):.3f}"
def paired(a, b):
    d = a - b; return f"{d.mean():+.3f} (SE {d.std(ddof=1)/np.sqrt(len(d)):.3f}, p={stats.ttest_rel(a, b).pvalue:.2g}, {int((d>0).sum())}/{len(d)} up)"

P(f"=== linear-algebra structure of the pool ({z['done']} modules; energy-weighted over modules; n={len(keep)} tasks, duplicates once) ===")
P("residual = D_t - P16(D_t) (operator form); residual energy share of full delta:",
  m(rn2[keep] / dn2[keep]), f"(range {np.min(rn2[keep]/dn2[keep]):.3f}-{np.max(rn2[keep]/dn2[keep]):.3f})")

P("\n--- EXP-88a: is there a SECOND shared component in the residuals? stacked-residual subspace, 4-fold held-out ---")
P(f"{'k':>4} {'held-out':>9} {'held-in':>8} {'out/in':>7} {'chance':>9} {'left-only held-out':>19}")
for i, k in enumerate(KR):
    ho, hi, ch, hl = z["st_ho"][keep, i] / rn2[keep], z["st_hi"][keep, i] / rn2[keep], z["chance"][keep, i] / rn2[keep], z["st_ho_left"][keep, i] / rn2[keep]
    P(f"{k:>4} {ho.mean():>9.4f} {hi.mean():>8.4f} {ho.mean()/hi.mean():>7.2f} {ch.mean():>9.2e} {hl.mean():>19.4f}")
spec = z["stack_spec"] / z["stack_tot"]
P("full-76 stacked residual spectrum (left, one-sided): top-k share of all residual energy:",
  {k: round(float(spec[:k].sum()), 3) for k in (1, 4, 16, 32, 64)})
P("reference: S16 captures 0.80 of the FULL deltas, held-out = held-in (the first shared component)")

P("\n--- EXP-88b: cluster-level shared subspace (semantic clusters) vs random groups of equal size ---")
for kc in (5, 8):
    lab = np.array(meta.get("clusters", {}).get(str(kc), [0] * N))
    sizes = np.bincount(lab)[1:] if len(lab) else []
    P(f"k={kc} clusters, sizes {sorted(sizes.tolist(), reverse=True) if len(sizes) else '?'}")
    for i, k in enumerate(KR):
        c = z[f"r_clu{kc}"][keep, i] / rg[keep]; r = z[f"r_rclu{kc}"][keep, i] / rg[keep]
        ok = c > 0
        P(f"   rank {k:>2}: cluster {c[ok].mean():.4f}  random {r[ok].mean():.4f}  ratio {c[ok].mean()/max(r[ok].mean(),1e-12):.2f}  paired {paired(c[ok], r[ok])}")
ipR = z["ipR"]
def vec_cos(s, g):
    g = [x for x in g if x not in (s, DUP.get(s, -1))]
    if not g: return np.nan
    return ipR[s, g].sum() / np.sqrt(ipR[s, s] * ipR[np.ix_(g, g)].sum())
rng = np.random.default_rng(1)
for kc in (5, 8):
    lab = np.array(meta.get("clusters", {}).get(str(kc), [0] * N))
    cc, rr = [], []
    for s in keep:
        g = [x for x in range(N) if lab[x] == lab[s]]
        others = [x for x in range(N) if x not in (s, DUP.get(s, -1))]
        cc.append(vec_cos(s, g)); rr.append(np.nanmean([vec_cos(s, rng.choice(others, min(len(others), max(1, len(g) - 1)), replace=False).tolist()) for _ in range(20)]))
    cc, rr = np.array(cc), np.array(rr); ok = ~np.isnan(cc)
    P(f"cluster-MEAN residual as a shared VECTOR (k={kc}): cos(r_t, mean of its cluster) {cc[ok].mean():+.3f} vs random {rr[ok].mean():+.3f}"
      f" -> energy share cos^2 {np.mean(cc[ok]**2):.4f}")

P("\n--- EXP-89: does a target share more with SOME tasks? nearest residual neighbours vs random ---")
P("(a) residual subspace of n neighbours: capture of the target's residual")
for n in NN:
    row = []
    for i, k in enumerate(KR):
        a_ = z[f"r_nn{n}"][keep, i] / rg[keep]; b_ = z[f"r_rnd{n}"][keep, i] / rg[keep]
        row.append(f"k{k}: nn {a_.mean():.4f} / rnd {b_.mean():.4f} (x{a_.mean()/b_.mean():.2f})")
    P(f"  n={n:>2}: " + " | ".join(row))
pool = z["core16_g"] / dg_
P(f"(b) soup of n tasks' FULL deltas: capture of the target's FULL delta (pool S16 = {pool[keep].mean():.4f})")
for n in NN:
    for i, k in enumerate(KS):
        a_ = z[f"s_nn{n}"][keep, i] / dg_[keep]; b_ = z[f"s_rnd{n}"][keep, i] / dg_[keep]
        P(f"  n={n:>2} rank {k}: neighbours {a_.mean():.4f}  random {b_.mean():.4f}  pool@16 {pool[keep].mean():.4f}   nn-rnd {paired(a_, b_)}   nn-pool {paired(a_, pool[keep])}")
for kc in (5, 8):
    for i, k in enumerate(KS):
        a_ = z[f"s_clu{kc}"][keep, i] / dg_[keep]; b_ = z[f"s_rclu{kc}"][keep, i] / dg_[keep]; ok = a_ > 0
        P(f"  cluster k={kc} rank {k}: cluster soup {a_[ok].mean():.4f}  random {b_[ok].mean():.4f}   {paired(a_[ok], b_[ok])}")

P("\n--- EXP-90: per-task use of the shared component (cores C_t = U16' D_t V16) ---")
cur = z["core_curve"] / dn2[:, None]
P("capture of own delta by pool S_k, across tasks: k: mean / sd / min / max")
for k in (1, 2, 4, 8, 12, 16):
    c = cur[keep, k - 1]; P(f"  k={k:>2}: {c.mean():.3f} / {c.std():.3f} / {c.min():.3f} (t{keep[np.argmin(c)]}) / {c.max():.3f} (t{keep[np.argmax(c)]})")
tail = (cur[:, 15] - cur[:, 7]) / cur[:, 15]
P(f"tail share (directions 9-16 of the task's shared energy): mean {tail[keep].mean():.3f} sd {tail[keep].std():.3f} range {tail[keep].min():.3f}-{tail[keep].max():.3f}")
if base is not None:
    P(f"  corr(tail share, own baseline) = {stats.pearsonr(tail[keep], base[keep])[0]:+.3f} (p={stats.pearsonr(tail[keep], base[keep])[1]:.2g});"
      f" corr(capture@16, baseline) = {stats.pearsonr(cur[keep,15], base[keep])[0]:+.3f}")
row = z["core_row"] / z["core_row"].sum(1, keepdims=True)
P("per-direction loading (share of the task's core energy on shared direction i), across tasks: CV per direction:",
  [round(float(x), 2) for x in (row[keep].std(0) / row[keep].mean(0))])
G = z["coreG"]; dg = np.sqrt(np.diag(G)); cosC = G / np.outer(dg, dg)
CG = z["coreCG"]; dc = np.sqrt(np.diag(CG)); cosCC = CG / np.outer(dc, dc)
iu = np.triu_indices(N, 1); mask = np.array([(i not in (18, 62) and j not in (18, 62)) for i, j in zip(*iu)])
P(f"cores: mean pairwise cosine {cosC[iu][mask].mean():.4f} (sd {cosC[iu][mask].std():.4f}) -> each task's shared part is ~the same")
P(f"task-specific part of the core (C_t - mean core): {np.mean(np.diag(CG)[keep]/np.diag(G)[keep]):.4f} of core energy;"
  f" of full delta: {np.mean(np.diag(CG)[keep]/dn2[keep]):.4f}")
Cres = np.load(REPO / "results/analysis/residual_cosines.npz")["C"]
P(f"centred-core cosines: mean {cosCC[iu][mask].mean():+.4f} sd {cosCC[iu][mask].std():.4f} max {cosCC[iu][mask].max():+.3f};"
  f" corr with residual cosines (off-diag) {np.corrcoef(cosCC[iu][mask], Cres[iu][mask])[0,1]:+.3f}")
nnC = [set(np.argsort(-np.where(np.isin(np.arange(N), [s, DUP.get(s, -1)]), -9, cosCC[s]))[:5]) for s in range(N)]
nnR = [set(np.argsort(-np.where(np.isin(np.arange(N), [s, DUP.get(s, -1)]), -9, Cres[s]))[:5]) for s in range(N)]
P(f"top-5 neighbours by centred core vs by residual: mean overlap {np.mean([len(nnC[s] & nnR[s]) for s in keep]):.2f}/5 (chance {25/73:.2f})")
txt = "\n".join(out); print(txt)
(REPO / f"results/analysis/{f.stem}_report.txt").write_text(txt + "\n")
