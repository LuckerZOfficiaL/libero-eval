#!/usr/bin/env python3
"""Turn a LoRA checkpoint into a tensor the HyperConv encoder can eat — RAW FACTORS, no features.

The network's input is the adapter itself: the B and A matrices as trained. Nothing is summarised,
correlated, or hand-designed (contrast the previous hypernetwork, which consumed residual cosines,
TF-IDF language vectors and effective ranks).

The only transformation is SHAPE. Our 458 modules have incompatible dimensions (1024, 2048, 4096, and
lm_head at 257152), while a convolutional stack needs one uniform tensor. Each module's factors are
therefore mapped to a common width by a FIXED random projection:

    B (d_out x 16)  -> P_out^T B  ->  (W x 16) -> transposed -> (16 x W)
    A (16 x d_in)   -> A P_in     ->  (16 x W)
    module tensor    = concat -> (32, W)                       [2*rank rows]
    adapter tensor   = stack over modules -> (M, 32, W)

The projections are drawn once from a fixed seed and SHARED by every adapter, so two adapters that are
close in weight space stay close after projection (Johnson-Lindenstrauss). They are not learned and
carry no information about the task — they are a shape adapter, not a feature extractor.

KNOWN LIMITATION, stated because it will be the first suspect if v1 fails to learn: LoRA factors are
gauge-ambiguous, B@A = (BR)(R^-1 A), so two IDENTICAL adapters can present different B and A. Raw
factors therefore contain arbitrary rotation noise the encoder must learn to ignore, from only 76
adapters. `--canon svd` swaps in the rotation-invariant canonical form (singular values and projected
singular directions of the same delta) as a one-flag ablation.

Module selection: the 458 stems are sorted by their energy contribution and the top M are kept, so the
tensor covers the modules that actually carry the adaptation rather than an arbitrary slice.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import safetensors.torch as st
import apply_lora as al

RANK = 16
WIDTH = 64          # common projected width
N_MOD = 128         # modules kept (of 458), by energy
SEED = 20260901


def load_factors(path):
    """{stem: (B, A)} with the LoRA scale already folded in — exactly as the merge operators use them."""
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(str(path))).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)
    return out


def choose_modules(paths, n_mod=N_MOD):
    """Rank stems by mean delta energy across a sample of adapters; keep the top n_mod."""
    energy = {}
    for p in paths:
        for stem, (b, a) in load_factors(p).items():
            e = float((b @ a).pow(2).sum())
            energy[stem] = energy.get(stem, 0.0) + e
    stems = sorted(energy, key=energy.get, reverse=True)[:n_mod]
    return sorted(stems)


class Tokenizer:
    def __init__(self, stems, width=WIDTH, seed=SEED, canon="raw"):
        self.stems = list(stems)
        self.width = width
        self.canon = canon
        self.g = torch.Generator().manual_seed(seed)
        self._proj = {}                      # (dim, tag) -> fixed projection matrix

    def _P(self, dim, tag):
        key = (dim, tag)
        if key not in self._proj:
            # fixed JL projection, scaled to preserve norms in expectation
            self._proj[key] = torch.randn(dim, self.width, generator=self.g) / (self.width ** 0.5)
        return self._proj[key]

    def encode(self, path_or_folds):
        f = load_factors(path_or_folds) if not isinstance(path_or_folds, dict) else path_or_folds
        rows = []
        for stem in self.stems:
            if stem not in f:
                rows.append(torch.zeros(2 * RANK, self.width))
                continue
            b, a = f[stem]                                  # b: (d_out, r), a: (r, d_in)
            r = b.shape[1]
            if self.canon == "svd":
                # rotation-invariant canonical form of the SAME delta: singular values scale the
                # projected singular directions, so the gauge freedom R is removed.
                qb, rb = torch.linalg.qr(b, mode="reduced")
                qa, ra = torch.linalg.qr(a.T, mode="reduced")
                u, s, vt = torch.linalg.svd(rb @ ra.T, full_matrices=False)
                left = (qb @ u) * s                          # (d_out, r)
                right = (qa @ vt.T) * s                      # (d_in, r)
                bb = (self._P(left.shape[0], "out").T @ left).T
                aa = (self._P(right.shape[0], "in").T @ right).T
            else:
                # RAW FACTORS: the matrices as trained, only projected to a common width.
                bb = (self._P(b.shape[0], "out").T @ b).T     # (r, W)
                aa = a @ self._P(a.shape[1], "in")            # (r, W)
            row = torch.cat([bb, aa], 0)                      # (2r, W)
            if row.shape[0] < 2 * RANK:
                row = torch.cat([row, torch.zeros(2 * RANK - row.shape[0], self.width)], 0)
            rows.append(row[: 2 * RANK])
        x = torch.stack(rows, 0)                              # (M, 2*RANK, W)
        # per-adapter standardisation: adapters differ in overall scale, which is a nuisance here
        # because the coefficient head should react to STRUCTURE, not to how large a delta happens to be
        return (x - x.mean()) / (x.std() + 1e-6)


def build(canon="raw", n_mod=N_MOD, out=None, plan_file="sweep/plan.json"):
    plan = json.load(open(REPO / plan_file))["adapters"]
    paths = {a["task_index"]: a["adapter_file"] for a in plan}
    stems = choose_modules([paths[t] for t in (0, 6, 19, 50)], n_mod)
    tok = Tokenizer(stems, canon=canon)
    X = torch.stack([tok.encode(paths[t]) for t in sorted(paths)])
    out = out or REPO / f"results/analysis/lora_tokens_{canon}.pt"
    torch.save({"X": X, "stems": stems, "canon": canon, "tasks": sorted(paths)}, out)
    print(f"{canon}: {tuple(X.shape)}  (adapters, modules, 2*rank, width)  -> {out}")
    return X


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", default="raw", choices=["raw", "svd"])
    ap.add_argument("--n_mod", type=int, default=N_MOD)
    ap.add_argument("--plan", default="sweep/plan.json", help="plan106.json adds the 30 S16-init experts; module choice stays fleet-derived so the first 76 rows are unchanged")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    X = build(a.canon, a.n_mod, out=a.out, plan_file=a.plan)
    # sanity: do different adapters actually look different, and do identical ones match?
    d01 = (X[0] - X[1]).abs().mean()
    d00 = (X[0] - X[0]).abs().mean()
    print(f"  |t0 - t1| = {d01:.4f}   |t0 - t0| = {d00:.4f}")
    print(f"  duplicate check t7 vs t18 (identical md5): {(X[7] - X[18]).abs().mean():.6f} (want ~0)")
