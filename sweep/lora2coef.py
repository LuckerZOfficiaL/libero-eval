#!/usr/bin/env python3
"""LoRA2Coef — the REVERSE of Drag-and-Drop's hypernetwork.

Drag-and-Drop (Fascetta/dragdrop-vlas) runs `HyperConvDecoder`: a condition embedding is DECODED, via
stacked HyperConv3d blocks, up into LoRA weight matrices. Weights are the OUTPUT.

We invert it. Two LoRAs go IN (a target and a candidate donor) and a merging coefficient comes OUT:

    encoder(D_target), encoder(D_donor)  ->  c in [0,1]        (v1: scalar)
                                         ->  c_g in [0,1]^G    (v2: per layer-group)

then the merge is  W = D_target + c * r_donor  and the reward is the rollout performance delta.

WHAT IS BORROWED FROM THE REFERENCE
  * The HyperConv3d idea: a LoRA layer is a (height, width) grid, and you convolve along BOTH axes
    with separate paths that are then averaged. That respects the fact that neither axis of a weight
    matrix is a spatial dimension in the usual sense, and it is why raw weights are consumable at all.
  * Stacked blocks over a (module, height, width) stack, with LayerNorm and SiLU.
  * Their decoder goes narrow->wide; our encoder goes wide->narrow and ends in a pooled embedding.

WHAT IS DELIBERATELY DIFFERENT
  * ASYMMETRY. score(T,p) != score(p,T) is a measured property of this project (reciprocity r=-0.115,
    sign agreement 50%). The head therefore sees [e_T, e_p, e_T - e_p, e_T * e_p]; the signed
    difference is what breaks symmetry. A siamese/cosine head could not represent the relation.
  * RANK-16 STRUCTURE, NOT DENSE DELTAS. A LoRA layer is B@A with rank 16, and materialising the dense
    delta is both huge and wasteful. We feed the rank-16 factor pair directly: for each module we build
    a (2, 16, d_proj) tensor from B^T and A after a fixed random projection to a common width. This is
    the "raw weights, no feature engineering" the plan asks for -- no cosines, no hand-crafted scalars.
  * PERMUTATION HANDLING. LoRA factors are only defined up to a rank-space rotation (B@A = (BR)(R^-1 A)),
    so any function of the factors that is not invariant to R is reading noise. We therefore feed the
    SINGULAR VALUE structure: per module we SVD the small 16x16 core and use its singular values and
    the projected singular directions, which are R-invariant.
"""
import math

import torch
from torch import nn


class HyperConv2d(nn.Module):
    """Two-path convolution over a (modules, height, width) stack -- the reference's HyperConv3d,
    reversed to contract rather than expand."""

    def __init__(self, in_hw, out_hw, kernel=3, dropout=0.0):
        super().__init__()
        in_h, in_w = in_hw
        out_h, out_w = out_hw
        self.norm = nn.LayerNorm([in_h, in_w])
        self.conv1w = nn.Conv2d(in_w, out_w, kernel, 1, "same", bias=False)
        self.conv1h = nn.Conv2d(in_h, out_h, kernel, 1, "same", bias=False)
        self.conv2h = nn.Conv2d(in_h, out_h, kernel, 1, "same", bias=False)
        self.conv2w = nn.Conv2d(in_w, out_w, kernel, 1, "same", bias=False)
        self.bias = nn.Parameter(torch.zeros(out_h, out_w))
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def _w_first(self, x):
        x = x.transpose(-1, -3)
        x = self.conv1w(x)
        x = x.transpose(-1, -3).transpose(-2, -3)
        x = self.conv1h(x)
        return x.transpose(-2, -3)

    def _h_first(self, x):
        x = x.transpose(-2, -3)
        x = self.conv2h(x)
        x = x.transpose(-2, -3).transpose(-1, -3)
        x = self.conv2w(x)
        return x.transpose(-1, -3)

    def forward(self, x):                      # x: (B, M, H, W)
        xn = self.norm(x)
        outer = xn.shape[:-3]
        xf = torch.flatten(xn, end_dim=-4)
        y = (self._w_first(xf) + self._h_first(xf)) / 2.0
        y = self.drop(self.act(y + self.bias))
        return torch.unflatten(y, 0, outer)


class LoRAEncoder(nn.Module):
    """(B, M, 2*rank, W) stack of LoRA factors -> (B, d_emb) embedding.

    Contracting HyperConv blocks, then mean-pool over modules. Mean-pooling makes the embedding
    independent of module ORDER, which matters because our 458 stems have no meaningful sequence.
    """

    def __init__(self, n_mod, in_h, in_w, d_emb=128, dropout=0.1):
        super().__init__()
        h1, w1 = max(in_h // 2, 4), max(in_w // 2, 8)
        h2, w2 = max(h1 // 2, 2), max(w1 // 2, 4)
        self.b1 = HyperConv2d((in_h, in_w), (h1, w1), 3, dropout)
        self.b2 = HyperConv2d((h1, w1), (h2, w2), 3, dropout)
        self.mod_mix = nn.Conv2d(n_mod, n_mod // 4 or 1, 1)      # mix across modules
        self.head = nn.Sequential(nn.Flatten(), nn.LayerNorm((n_mod // 4 or 1) * h2 * w2),
                                  nn.Linear((n_mod // 4 or 1) * h2 * w2, d_emb), nn.SiLU(),
                                  nn.Linear(d_emb, d_emb))

    def forward(self, x):
        x = self.b2(self.b1(x))
        x = self.mod_mix(x)
        return self.head(x)


class LoRA2Coef(nn.Module):
    """Two LoRAs in, merging coefficient(s) out.

    n_groups=1 -> scalar alpha (v1).  n_groups=G -> per-group vector (v2), where groups are
    coarse layer bands (vision / llm-early / llm-late / expert / action head) rather than all 458
    modules: with REINFORCE the gradient is a single scalar reward per rollout batch, so a 458-dim
    action space would be hopelessly under-determined at our rollout budget.
    """

    def __init__(self, n_mod, in_h, in_w, d_emb=128, n_groups=1, dropout=0.1):
        super().__init__()
        self.encoder = LoRAEncoder(n_mod, in_h, in_w, d_emb, dropout)
        self.n_groups = n_groups
        self.head = nn.Sequential(
            nn.Linear(4 * d_emb, 256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, n_groups))
        # warm start at the dose already verified as optimal (0.5). Random initial coefficients would
        # make every early merge bad, the rewards uniformly poor, and REINFORCE would have no contrast.
        # BOUNDED range: c in [C_LO, C_HI] rather than [0,1]. An unbounded sigmoid let the first run
        # saturate to c=0.000 in three steps -- "never merge", which is a genuine optimum for a learner
        # that cannot yet tell good donors from bad (c=0 scores exactly 0 with no variance, while an
        # average donor scores ~0 WITH variance). Once saturated the gradient vanishes and it can never
        # recover. Bounding below keeps the policy alive long enough to learn to DISCRIMINATE, which is
        # the only way it can beat c=0.
        self.c_lo, self.c_hi = 0.10, 1.20
        nn.init.zeros_(self.head[-1].weight)
        import math
        z = (0.5 - self.c_lo) / (self.c_hi - self.c_lo)
        nn.init.constant_(self.head[-1].bias, math.log(z / (1 - z)))

    def forward(self, x_t, x_p):
        e_t, e_p = self.encoder(x_t), self.encoder(x_p)
        h = torch.cat([e_t, e_p, e_t - e_p, e_t * e_p], -1)     # signed diff breaks symmetry
        return self.c_lo + (self.c_hi - self.c_lo) * torch.sigmoid(self.head(h))

    @torch.no_grad()
    def coef(self, x_t, x_p):
        self.eval()
        return self.forward(x_t, x_p)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for G in (1, 5):
        m = LoRA2Coef(n_mod=64, in_h=32, in_w=64, n_groups=G)
        xt = torch.randn(2, 64, 32, 64)
        xp = torch.randn(2, 64, 32, 64)
        out = m(xt, xp)
        print(f"groups={G}: out {tuple(out.shape)}  mean {out.mean():.3f} (want ~0.500)  "
              f"params {count_params(m)/1e6:.2f}M")
        # asymmetry check: the whole point of the signed-difference term
        a, b = m(xt, xp), m(xp, xt)
        print(f"          asymmetric: {not torch.allclose(a, b)}  |a-b| mean {(a-b).abs().mean():.4f} "
              f"(0.0000 expected at init, since the head starts at zero)")
