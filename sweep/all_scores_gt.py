#!/usr/bin/env python3
"""All 130 donor scores (categories A-F) vs the EXP-95 ground truth, one consistent evaluation.
Within-receiver Spearman between the raw score and ground-truth success over all 73 donors; reported signed,
summarised per category as best |rho| and median |rho|, for weak receivers (25-rollout GT, ceiling 0.80) and all
74 (ceiling 0.52). D (sensitivity) exists for the 15 weak receivers only."""
import glob, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import gt_analysis as A_, la_combo as C
import unified_pool_driver as U
from subset_probe import TARGETS as WEAK
BANDS = ["all", "vision", "llm_early", "llm_late", "expert", "head"]

def cat_A():
    F = {k: v for k, v in C.la_feats().items() if not k.startswith("IN_")}          # 60 = #1-10 x 6
    z = np.load(REPO / "results/analysis/linalg_structure.npz"); CG = z["coreCG"]; dc = np.sqrt(np.diag(CG))
    F["centred_core_cos|all"] = CG / np.outer(dc, dc)                                   # #11
    return F
def cat_B():
    zi = np.load(REPO / "results/analysis/la_input_score.npz")
    mag, dr, rn = [np.concatenate([zi[k].sum(0, keepdims=True), zi[k]], 0) for k in ("mag", "dir", "rn2")]
    F = {}
    for b in range(6):
        self_ = np.array([mag[b, t, t] for t in range(76)])[:, None]
        g = mag[b] / np.maximum(rn[b][None, :], 1e-30)
        F[f"IN_gain|{BANDS[b]}"] = g
        F[f"IN_align|{BANDS[b]}"] = dr[b] / np.sqrt(np.maximum(mag[b] * self_, 1e-60))
        F[f"IN_rel_gain|{BANDS[b]}"] = g / np.maximum(self_ / np.maximum(rn[b][:, None], 1e-30), 1e-30)
    return F
def merge(pat, keys):
    out = {}
    for f in sorted(glob.glob(str(REPO / f"results/analysis/{pat}"))):
        z = np.load(f)
        for k in keys:
            x = z[k].astype(float)[:, :76]
            out[k] = x.copy() if k not in out else np.where(np.isnan(out[k]), x, out[k])
    return out
def cat_C(): return merge("energy_proxy_s*.npz", ["disp", "drift", "demo", "ddisp"])
def cat_D(): return merge("sens_proxy_s*.npz", ["sens_h", "sens_v", "sens_img", "rel_img", "rel_h"])
def cat_E():
    sys.path.insert(0, str(REPO / "sweep")); import proxy_eval as PE
    return {k: v[1] for k, v in PE.grad_scores().items()}
def cat_F():
    z = np.load(REPO / "results/analysis/ties_conflict.npz")
    def ag(k): x = z[k]; return np.concatenate([x.sum(0, keepdims=True), x], 0)
    F = {}
    for b in range(6):
        F[f"conflict|{BANDS[b]}"] = ag("conf_n")[b] / np.maximum(ag("conf_d")[b], 1e-9)
        F[f"overlap|{BANDS[b]}"] = ag("ovl_n")[b] / np.maximum(ag("ovl_d")[b], 1e-9)
        F[f"agree_all|{BANDS[b]}"] = ag("agr_n")[b] / np.maximum(ag("agr_d")[b], 1e-9)
    return F

def main():
    GT = {t: {d: A_.rec(t, d)["success_rate"] for d in A_.UNIQUE if d != t} for t in A_.UNIQUE}
    cats = [("A weights only", cat_A()), ("B weights x target inputs", cat_B()), ("C label-free energy", cat_C()),
            ("D image-shift sensitivity", cat_D()), ("E gradients + Fisher", cat_E()), ("F TIES sign conflicts", cat_F())]
    L = ["=== all donor scores vs EXP-95 ground truth (within-receiver Spearman, signed) ==="]
    total = 0; allrows = []
    for cname, F in cats:
        rows = []
        for nm, M in F.items():
            r = {}
            for grp, T in (("weak", [t for t in A_.UNIQUE if t in WEAK]), ("all", A_.UNIQUE)):
                rr = []
                for t in T:
                    ds = sorted(GT[t]); y = np.array([GT[t][d] for d in ds]); x = np.asarray(M[t, ds], float)
                    if np.isnan(x).all() or y.std() == 0 or np.nanstd(x) == 0: continue
                    rr.append(stats.spearmanr(x, y, nan_policy="omit").statistic)
                r[grp] = (np.mean(rr), len(rr)) if rr else (np.nan, 0)
            rows.append((nm, r))
        total += len(rows); allrows += [(cname, nm, r) for nm, r in rows]
        w = np.array([abs(r["weak"][0]) for _, r in rows]); a = np.array([abs(r["all"][0]) for _, r in rows if r["all"][1] > 15])
        bw = max(rows, key=lambda x: abs(x[1]["weak"][0]))
        L.append(f"{cname:<28} {len(rows):>3} scores | weak: best {bw[1]['weak'][0]:+.3f} ({bw[0]}), median |rho| {np.median(w):.3f}"
                 + (f" | all: best |rho| {a.max():.3f}, median {np.median(a):.3f}" if len(a) else " | all: n/a (weak receivers only)"))
    w = np.array([abs(r["weak"][0]) for _, _, r in allrows])
    L.append(f"TOTAL {total} scores | weak: best |rho| {w.max():.3f}, median {np.median(w):.3f}, {int((w >= 0.1).sum())} scores >= 0.10 | ceiling 0.80")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/all_scores_gt_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
