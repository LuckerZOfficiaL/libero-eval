#!/usr/bin/env python3
"""EXP-34 — do RAW WEIGHTS beat hand-crafted features, holding the training signal fixed?

Three attempts have now failed to learn the (target, donor) interaction:
  1. supervised on 43 hand features (residual-Gram / language / scalars)  -> holdout gain +1.31pp
  2. the same, deployed with real rollouts                                -> -7.0pp vs the probe
  3. REINFORCE on raw weights                                             -> donor-independent constant
     (coefficient for known-GOOD and known-BAD donors identical to 4 decimals)

Attempt 3 changed TWO things at once versus attempt 1: the representation (raw weights instead of hand
features) AND the training signal (noisy policy gradients instead of 634 precomputed labels). That is a
confounded comparison, and the confound matters because the two failures have different remedies --
more data fixes one, more compute fixes the other.

This isolates the representation. Same 634 rollout-derived labels, same binary "is this donor good"
objective, same leave-out-10-targets protocol as the earlier study. The ONLY change is that the pair
embedding comes from the HyperConv encoder reading raw B/A factors instead of the 43-d feature vector.

  raw weights better than +1.31pp -> representation was a real limit; REINFORCE deserves more budget
  raw weights ~= or worse         -> the representation was never the bottleneck, and no amount of
                                     rollout budget fixes a signal that is not in the weights

Costs nothing but CPU: the labels and the tokenised weights already exist.
"""
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep"))
from lora2coef import LoRAEncoder
import hypernet as H

torch.set_num_threads(4)


class RawPairNet(torch.nn.Module):
    """Identical head geometry to the feature-based study; only the embedding source differs."""

    def __init__(self, n_mod, in_h, in_w, d_emb=64, drop=0.1):
        super().__init__()
        self.enc = LoRAEncoder(n_mod, in_h, in_w, d_emb=d_emb, dropout=drop)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(4 * d_emb, 64), torch.nn.GELU(), torch.nn.Dropout(drop),
            torch.nn.Linear(64, 1))

    def forward(self, xt, xp):
        a, b = self.enc(xt), self.enc(xp)
        return self.head(torch.cat([a, b, a - b, a * b], -1)).squeeze(-1)


def main():
    import argparse
    from scipy import stats
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="results/analysis/lora_tokens_raw.pt")
    ap.add_argument("--labels", default="results/analysis/pair_labels.json")
    ap.add_argument("--feats", default="results/analysis/hypernet_feats.npz")
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--holdout", default="random", choices=["random", "fleet", "new"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="results/analysis/raw_supervised_report.txt")
    a = ap.parse_args()
    tok = torch.load(REPO / a.tokens)
    X, tasks = tok["X"], list(tok["tasks"])
    idx = {t: i for i, t in enumerate(tasks)}
    _, groups, baseline = H.load(str(REPO / a.labels), feats=a.feats, baselines=a.baselines)
    print(f"[raw-sup] {len(groups)} target groups, {sum(len(g[1]) for g in groups)} labels, "
          f"tokens {tuple(X.shape)}", flush=True)

    gains = []
    for seed in range(a.seeds):
        tr, te, ho = H.split(groups, baseline, seed=seed, mode=a.holdout)
        torch.manual_seed(seed); np.random.seed(seed)
        net = RawPairNet(X.shape[1], X.shape[2], X.shape[3])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(40):
            net.train()
            for i in np.random.permutation(len(tr)):
                T, cand, aff, lw = tr[i]
                s = net(X[idx[T]].unsqueeze(0).expand(len(cand), -1, -1, -1),
                        X[[idx[c] for c in cand]])
                loss = H.binary_loss(s, aff, lw)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()

        def score(T, cand):
            with torch.no_grad():
                return net(X[idx[T]].unsqueeze(0).expand(len(cand), -1, -1, -1),
                           X[[idx[c] for c in cand]]).numpy()

        m = H.evaluate(score, te)
        gains.append(m["topn"] - m["random"])
        print(f"  seed {seed}: top-3 gain over random {gains[-1]:+.2f}pp  "
              f"(spearman {m['spearman']:+.3f}, score sd {m['spread']:.3f})", flush=True)

    g = np.array(gains)
    t = stats.ttest_1samp(g, 0)
    L = [f"=== raw LoRA weights (HyperConv encoder), labels={a.labels.split('/')[-1]} holdout={a.holdout} seeds={a.seeds} ===",
         "634 rollout-derived labels, binary 'is this donor good', leave-out-10-targets, 8 seeds", "",
         f"  RAW WEIGHTS  : {g.mean():+.2f}pp over random  (SE {stats.sem(g):.2f}, t={t.statistic:+.2f}, "
         f"p={t.pvalue:.3f})",
         f"  hand features: +1.31pp (p = 0.0065)   <- the earlier study, same protocol",
         f"  per-seed: {np.round(g, 2).tolist()}", "",
         "  raw >> features -> representation was the limit; more rollout budget is justified",
         "  raw <= features -> representation was never the limit, and the interaction is simply not",
         "                     recoverable from 76 adapters by any of these routes"]
    open(REPO / a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
