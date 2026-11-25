#!/usr/bin/env python3
"""EXP-103c: matched-state READOUT CONFLICT (input similarity x action agreement) -- data only, no merges, no rollouts.

Theory (Li et al. ICLR 2025 "When is task vector provably effective"; Hiratani NeurIPS 2024; Evron et al. ICLR 2024):
adding a task vector helps for ALIGNED tasks (similar inputs, same outputs), is neutral for IRRELEVANT ones
(different inputs) and hurts for CONTRADICTORY ones (similar inputs, different outputs). Similarity alone mixes
the first and the last -- which could be why every similarity metric so far is ~0 or slightly negative.

Per task: frames from 10 demos (stride 4), both cameras embedded by the BASE SigLIP tower (mean-pooled tokens,
L2-normalised, cameras concatenated); expert action chunk (quantile-normalised, 7 dims).
Per (target t, donor d): for each target frame i, its best-matching donor frame j*(i) (cosine s_i):
  S_in     mean_i s_i                              input similarity
  agree    -mean_i ||a_i - a_j*||                  action agreement at matched states
  conflict mean_i s_i * ||a_i - a_j*||             similar state, different action (predicted HARMFUL)
  agree_hi agreement over the target frames whose match is in the top quartile of s (the states the donor "sees")
  first    the same as agree on the first action of the chunk only
Output results/analysis/exp103_readout.npz; evaluation in exp103_readout_eval (bottom of this file).
"""
import pathlib, sys, time
import numpy as np, torch
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
from reinforce_train import scenes as SC
from train_lora import load_demo_samples
BASE_DIR = pathlib.Path("/media/pinas/Chris/models/pi05_base_pytorch")
OUT = REPO / "results/analysis/exp103_readout_c.npz"                          # centred embeddings
N = 76

def embed_all(dev):
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(str(BASE_DIR), norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    pw = policy._model.paligemma_with_expert; policy._model.eval()
    E, A, ok = [], [], []
    for t in range(N):
        try:
            smp = load_demo_samples("libero_90_no_noops", SC[t][0], n_demos=10, stride=4)
        except FileNotFoundError:
            E.append(None); A.append(None); ok.append(False); print(f"[ro] t{t}: no demos", flush=True); continue
        embs = []
        for cam in ("observation/image", "observation/wrist_image"):
            x = np.stack([o[cam] for o, _ in smp])                                    # [n, 224, 224, 3] uint8
            out = []
            for i in range(0, len(x), 64):
                xb = torch.from_numpy(x[i:i + 64]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1.0
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    f = pw.embed_image(xb).float().mean(1)
                out.append(f.cpu())
            embs.append(torch.cat(out))
        E.append(torch.stack(embs, 1).half().numpy())                                 # [n, 2 cams, dim] raw mean-pooled
        A.append(np.stack([c[:, :7] for _, c in smp]).astype(np.float32)); ok.append(True)
        print(f"[ro] t{t}: {len(smp)} frames", flush=True)
    # centre per camera by the global mean frame (mean-pooled SigLIP features share a dominant common component:
    # raw best-match cosines were all > 0.99), then L2-normalise per camera and concatenate
    raw = np.concatenate([e for e in E if e is not None]).astype(np.float32)
    mu = raw.mean(0, keepdims=True)
    np.savez(REPO / "results/analysis/exp103_frames.npz", mu=mu, **{f"E{t}": e for t, e in enumerate(E) if e is not None},
             **{f"A{t}": a for t, a in enumerate(A) if a is not None})
    def fin(e):
        x = torch.from_numpy(e.astype(np.float32) - mu)
        return (torch.nn.functional.normalize(x, dim=2).flatten(1) / 2 ** 0.5).half().numpy()
    E = [fin(e) if e is not None else None for e in E]
    return E, A, ok

def pair_scores(E, A, ok, dev):
    K = ("S_in", "agree", "conflict", "agree_hi", "first")
    F = {k: np.full((N, N), np.nan) for k in K}
    Et = [torch.from_numpy(e).to(dev).float() if e is not None else None for e in E]
    At = [torch.from_numpy(a).to(dev) if a is not None else None for a in A]
    # global top-quartile threshold of best-match similarity
    allbest = []
    for t in range(N):
        for d in range(N):
            if d == t or not (ok[t] and ok[d]): continue
            s, j = (Et[t] @ Et[d].T).max(1)
            dist = (At[t] - At[d][j]).flatten(1).norm(dim=1); d1 = (At[t][:, 0] - At[d][j][:, 0]).norm(dim=1)
            F["S_in"][t, d] = s.mean().item(); F["agree"][t, d] = -dist.mean().item()
            F["conflict"][t, d] = (s * dist).mean().item(); F["first"][t, d] = -d1.mean().item()
            allbest.append((t, d, s.cpu(), dist.cpu()))
    thr = float(np.quantile(torch.cat([x[2] for x in allbest]).numpy(), 0.75))
    for t, d, s, dist in allbest:
        m = s >= thr
        F["agree_hi"][t, d] = -dist[m].mean().item() if m.sum() >= 5 else np.nan
    F["thr"] = np.array(thr)
    return F

def evaluate(F):
    import gt_analysis as A_
    from subset_probe import TARGETS as WEAK
    UN = A_.UNIQUE; WK = [t for t in UN if t in WEAK]
    GT = {t: {d: A_.rec(t, d)["success_rate"] for d in UN if d != t} for t in UN}
    L = [f"=== EXP-103c matched-state readout conflict vs EXP-95 ground truth (within-receiver Spearman) ===",
         f"top-quartile best-match similarity threshold {float(F['thr']):.3f}"]
    for k in ("S_in", "agree", "conflict", "agree_hi", "first"):
        line = f"{k:9s}"
        for grp, T in (("weak", WK), ("all", UN)):
            rr = []
            for t in T:
                ds = [d for d in sorted(GT[t]) if np.isfinite(F[k][t, d])]
                if len(ds) < 20: continue
                y = np.array([GT[t][d] for d in ds]); x = F[k][t, ds]
                if y.std() > 0 and x.std() > 0: rr.append(stats.spearmanr(x, y).statistic)
            rr = np.array(rr)
            line += f"   {grp} {rr.mean():+.3f} (p={stats.ttest_1samp(rr, 0).pvalue:.3f}, n={len(rr)})"
        L.append(line)
    # theory check: does the sign of similarity depend on agreement? split donors by agreement within receiver
    L.append("\ninteraction (weak): Spearman(S_in, success) among donors with ABOVE- vs BELOW-median agreement")
    for half in ("above", "below"):
        rr = []
        for t in WK:
            ds = [d for d in sorted(GT[t]) if np.isfinite(F["agree"][t, d])]
            ag = np.array([F["agree"][t, d] for d in ds]); med = np.median(ag)
            sel = [d for d, a in zip(ds, ag) if (a >= med) == (half == "above")]
            y = np.array([GT[t][d] for d in sel]); x = F["S_in"][t, sel]
            if y.std() > 0 and x.std() > 0: rr.append(stats.spearmanr(x, y).statistic)
        rr = np.array(rr); L.append(f"   agreement {half} median: rho(S_in) {rr.mean():+.3f} (p={stats.ttest_1samp(rr,0).pvalue:.3f}, n={len(rr)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp103_readout_c_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    t0 = time.time()
    if not OUT.exists():
        E, A, ok = embed_all(dev)
        F = pair_scores(E, A, ok, dev)
        np.savez(OUT, **F); print(f"[ro] scores saved ({(time.time()-t0)/60:.1f} min)", flush=True)
    if "--no-eval" not in sys.argv: evaluate(dict(np.load(OUT)))
    print("[ro] DONE", flush=True)
