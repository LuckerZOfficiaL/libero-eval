#!/usr/bin/env python3
"""EXP-83 step 3: the cross-attention hypernetwork that EMITS the merging coefficients.

Faithful to DA-MergeLoRA's shape (their eq. 5-7):
  query  q  = MLP(mean frozen-encoder embedding of a batch of the RECEIVER's own frames)   [d]
  keys   k  = Linear(donor component token) + stem embedding + band embedding              [K, r, d]
  weights w = softmax over DONORS of <q,k>/tau, per (stem, component)                      [K, r]
so every rank-1 component of every module gets its own distribution over the 68 donors, exactly as they
weight every column of dW over their source domains. A learned global scale maps the simplex output to
the merge magnitude (their weights sum to 1 per column; ours must also choose HOW MUCH to inject).
"""
import pathlib, sys
import numpy as np, torch, torch.nn as nn
REPO = pathlib.Path(__file__).resolve().parent.parent

class CoefHypernet(nn.Module):
    def __init__(self, d_q, n_stems, K, r, d=128, d_tok=64, tau=1.0, p_drop=0.1, max_scale=4.0,
                 init_mass=None):
        super().__init__()
        self.K, self.r, self.d, self.tau, self.max_scale = K, r, d, tau, max_scale
        self.q = nn.Sequential(nn.Linear(d_q, 256), nn.GELU(), nn.Dropout(p_drop),
                               nn.Linear(256, d), nn.LayerNorm(d))
        self.k = nn.Linear(d_tok, d)
        self.stem_emb = nn.Embedding(n_stems, d)
        self.band_emb = nn.Embedding(5, d)
        self.donor_emb = nn.Embedding(K, d)                 # their per-domain token
        # scale: how much total mass to inject, per stem. starts near the verified 0.5 total.
        self.log_scale = nn.Parameter(torch.zeros(n_stems))
        # A softmax over K donors spreads the mass, and donor RESIDUALS are near-orthogonal, so the
        # injected norm is m/sqrt(K_eff) -- not m. To start at the energy our operator actually uses
        # (0.289 x mean||r||, i.e. 3 donors at ref/3 in quadrature) the initial mass must be
        # 0.289*sqrt(K). At K=65 that is 2.33, where a mass of 0.5 is 5x under-dosed.
        if init_mass is None:
            init_mass = 1.0        # the dose is energy-matched below; this is a multiplier on it
        f = init_mass / max_scale
        self.init_mass = init_mass
        self.register_buffer("_init", torch.tensor(float(np.log(f / (1 - f)))))
        # filled in by the caller: per-donor residual norms and the operator's ref (0.5*mean||r||)
        self.register_buffer("rnorm", torch.ones(K))
        self.ref = 1.0

    def forward(self, qv, tok, stem_ix, band_ix):
        """qv [d_q]; tok [n_stems, K, r, d_tok]; stem_ix [n_stems]; band_ix [n_stems]
        returns w [n_stems, K*r] -- softmax over donors, scaled per stem."""
        q = self.q(qv)                                                   # [d]
        k = (self.k(tok)
             + self.stem_emb(stem_ix)[:, None, None, :]
             + self.band_emb(band_ix)[:, None, None, :]
             + self.donor_emb.weight[None, :, None, :])                  # [S, K, r, d]
        logits = (k @ q) / (self.tau * np.sqrt(self.d))                  # [S, K, r]
        share = torch.softmax(logits, dim=1)                             # over DONORS, sums to 1
        # ENERGY MATCHING. Our operator gives each of 3 donors c_i = ref/(3||r_i||), so every donor
        # contributes ref/3 of NORM regardless of its own size, and the total (residuals being
        # near-orthogonal) is ref/sqrt(3) = 0.577*ref. A bare softmax does not do this: as training
        # concentrates the shares, the injected energy GROWS -- at max-share 0.45 the top donor reached
        # coefficient 1.04 against the operator's 0.167, ~2-4x over-dosed, and drove policies to 0%.
        # So: divide by each donor's residual norm (size-invariance) and renormalise by the share
        # vector's L2 norm (concentration-invariance). The learned scale then moves the dose, not the
        # arithmetic.
        c = share / self.rnorm[None, :, None].clamp(min=1e-6)            # size-invariant per donor
        l2 = c.mul(self.rnorm[None, :, None]).pow(2).sum(1, keepdim=True).sqrt().clamp(min=1e-9)
        c = c * (0.5773502692 / l2)                                      # -> injected energy == ref/sqrt(3)
        scale = torch.sigmoid(self.log_scale + self._init) * self.max_scale
        w = c * self.ref * scale[:, None, None]
        return w.reshape(w.shape[0], self.K * self.r)

def load_tokens(stems, donor_keys, device="cpu", d_tok=64, seed=83):
    """Donor component tokens: JL projection of each rank-1 column direction of dW, per stem."""
    sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
    import unified_pool_driver as U
    from reinforce_train import group_of
    g = torch.Generator().manual_seed(seed); proj = {}
    def P(n):
        if n not in proj: proj[n] = torch.randn(n, d_tok, generator=g) / np.sqrt(d_tok)
        return proj[n]
    folds = [U.fold(k) for k in donor_keys]
    r = folds[0][stems[0]][0].shape[1]
    tok = torch.zeros(len(stems), len(donor_keys), r, d_tok)
    for si, s in enumerate(stems):
        for ki, f in enumerate(folds):
            b, a = f[s]
            cols = b.float() * torch.linalg.norm(a.float(), dim=1)[None, :]   # [d_out, r]
            tok[si, ki] = cols.T @ P(cols.shape[0])
    tok = (tok - tok.mean()) / (tok.std() + 1e-6)
    band = torch.tensor([group_of(s) for s in stems], dtype=torch.long)
    return tok.to(device), band.to(device)
