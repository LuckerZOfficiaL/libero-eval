#!/usr/bin/env python3
"""EXP-82 -- DA-MergeLoRA-style cross-attention SELECTOR vs our feature model, same labels, same folds.

DA-MergeLoRA (arXiv 2607.17467): query = frozen-encoder mean embedding of a batch of TARGET-DOMAIN images;
keys = LoRA dW columns (+ layer / sub-layer / domain tokens); cross-attention -> per-column merge weights;
their ablation says the target images carry the signal. Here the same architecture is used as a per-donor
SCORER on our 1,354 rollout labels, evaluated with hypernet.py's protocol (hold out whole targets;
top-3 affinity vs random), paired seed-for-seed against the W+A/48 feature model (AffinityNet).
  query : mean 'vision' activation of the BASE policy over the target's own 16 demo frames (demo_bank_base)
  keys  : lora_col_tokens.npz (458 stems x 16 columns, JL-64) + learned stem + band embeddings
  score : mean over the donor's tokens of <q, k>/sqrt(d)   (the column-level softmax, pooled to a donor)
variants: obs   = query from observations only (faithful)
          obs+t = query also sees the target's own column summary
          feat  = AffinityNet on hypernet_featsexta48 W+A blocks (reference, same seeds)
"""
import argparse, json, pathlib, sys, time, numpy as np, torch, torch.nn as nn
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep"))
import hypernet as H
from scipy import stats

def load_inputs(query_key):
    F, groups, base = H.load("results/analysis/pair_labels_ext.json", "results/analysis/hypernet_featsexta48.npz",
                             "results/analysis/baselines_ext.json")
    z = np.load(REPO / "results/analysis/lora_col_tokens.npz")
    tok = torch.tensor(z["tok"].astype(np.float32))                # [172, 458, 16, 64]
    # CPU budget: keep the 4 largest column directions per stem (by token norm) -> 458 x 4 tokens
    idx = tok.norm(dim=-1).topk(4, dim=-1).indices                    # [172, 458, 4]
    tok = torch.gather(tok, 2, idx[..., None].expand(-1, -1, -1, tok.shape[-1]))
    stem_band = torch.tensor(z["stem_band"].astype(np.int64))       # [458]
    q = np.load(REPO / "results/analysis/demo_bank_base.npz"); tg = np.load(REPO / "results/analysis/demo_bank.npz")["target"]
    assert len(tg) == len(q[query_key]), (len(tg), len(q[query_key]))
    A = q[query_key].astype(np.float32)                              # [n_obs, ...] per-frame activation
    A = A.reshape(A.shape[0], -1) if A.ndim == 2 else A.mean(1)      # pool patch tokens if present
    Q = np.zeros((172, A.shape[1]), np.float32)
    for t in range(172):
        m = tg == t
        if m.any(): Q[t] = A[m].mean(0)
    Q = (Q - Q.mean(0)) / (Q.std(0) + 1e-6)
    return F, groups, base, tok, stem_band, torch.tensor(Q)

class XAttnScorer(nn.Module):
    def __init__(self, d_q, d_tok=64, d=128, n_stem=458, use_target=False, p_drop=0.2):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d_q, 256), nn.GELU(), nn.Dropout(p_drop), nn.Linear(256, d), nn.LayerNorm(d))
        self.k = nn.Linear(d_tok, d); self.stem = nn.Embedding(n_stem, d); self.band = nn.Embedding(5, d)
        self.use_target = use_target
        if use_target: self.tq = nn.Sequential(nn.Linear(d_tok, d), nn.GELU(), nn.Linear(d, d))
        self.d = d
    def keys(self, tok, stem_band):                                   # tok [n, 458, 16, 64]
        n, S, R, _ = tok.shape
        k = self.k(tok) + self.stem.weight[None, :, None, :] + self.band(stem_band)[None, :, None, :]
        return k.reshape(n, S * R, self.d)
    def forward(self, qv, ttok, ctok, stem_band):                     # qv [d_q]; ttok [458,16,64]; ctok [n,458,16,64]
        q = self.q(qv)
        if self.use_target: q = q + self.tq(ttok.mean((0, 1)))
        K = self.keys(ctok, stem_band)                                # [n, T, d]
        return (K @ q) .mean(1) / np.sqrt(self.d)                     # [n]

def run(a):
    a.seeds, a.epochs = min(a.seeds, 12), min(a.epochs, 100)   # capped: CPU-only training on the P6000 host
    print(f"[e82] seeds={a.seeds} epochs={a.epochs} (capped for CPU budget)", flush=True)
    F, groups, base, tok, stem_band, Q = load_inputs(a.query)
    sl = json.loads(str(np.load(REPO / "results/analysis/hypernet_featsexta48.npz", allow_pickle=True)["slices"]))
    keep = np.concatenate([np.arange(*sl[b]) for b in "WA"]); Ff = F[:, keep]
    res = {v: {"fleet": [], "new": []} for v in ("obs", "obs+t", "feat")}
    t0 = time.time()
    for mode in ("fleet", "new"):
        for seed in range(a.seeds):
            tr, te, ho = H.split(groups, base, seed=seed, mode=mode)
            for var in res:
                torch.manual_seed(seed); np.random.seed(seed)
                if var == "feat":
                    net = H.AffinityNet(Ff.shape[1], d_emb=32, p_drop=0.3)
                    sc = lambda T, c: net(Ff[T].expand(len(c), -1), Ff[c])
                else:
                    net = XAttnScorer(Q.shape[1], use_target=(var == "obs+t"))
                    sc = lambda T, c: net(Q[T], tok[T], tok[c], stem_band)
                opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
                for ep in range(a.epochs):
                    net.train()
                    for i in np.random.permutation(len(tr)):
                        T, cand, aff, lw = tr[i]
                        loss = H.listwise_loss(sc(T, cand), aff, tau=8.0, weight=float(aff.std()))
                        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                net.eval()
                with torch.no_grad():
                    m = H.evaluate(lambda T, c: sc(T, c).numpy(), te)
                res[var][mode].append(m["topn"] - m["random"])
            print(f"[{mode} seed {seed}] " + "  ".join(f"{v}: {res[v][mode][-1]:+.2f}" for v in res) + f"   ({time.time()-t0:.0f}s)", flush=True)
    print("\n=== EXP-82: top-3 affinity gain over random (pp), mean over seeds ===")
    for mode in ("fleet", "new"):
        for v in res:
            x = np.array(res[v][mode]); print(f"  {mode:6s} {v:6s} {x.mean():+.2f} (SE {x.std(ddof=1)/np.sqrt(len(x)):.2f}, p={stats.ttest_1samp(x,0).pvalue:.3f})")
        d = np.array(res["obs"][mode]) - np.array(res["feat"][mode])
        print(f"  {mode:6s} obs - feat (paired): {d.mean():+.2f} (SE {d.std(ddof=1)/np.sqrt(len(d)):.2f}, p={stats.ttest_1samp(d,0).pvalue:.3f})")
    json.dump(res, open(REPO / "results/analysis/exp82_xattn.json", "w"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=36); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--query", default="vision")
    run(ap.parse_args())
