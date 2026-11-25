#!/usr/bin/env python3
"""EXP-83 step 2: the differentiable fused merge.

DA-MergeLoRA merges COLUMNS of the full dW. Here one dW is 3.6B params (7.2GB fp16) and there are 68
donors, so their exact formula needs ~492GB and cannot be run. We keep the low-rank form instead and
emit one coefficient per (donor, rank-1 component):

    dW_merged = sum_k sum_r  w[k,r] * b[k,:,r] a[k,r,:]          (K=68 donors, r=16 components)

which is applied WITHOUT ever materialising dW:

    h' = W h + B_all @ (w * (A_all @ h))                          B_all [d_out, K*16], A_all [K*16, d_in]

so the cost is one extra pair of matmuls at effective rank 1088, and w is a plain tensor the gradient
flows through. All our adapters are rank 16, so the paper's stated reason for full-dW columns (mixing
different ranks) does not apply.
"""
import pathlib, sys
import torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U

class DonorBank:
    """A_all / B_all per stem, stacked once over the donor pool and kept on device."""
    def __init__(self, donor_keys, device="cuda:0", dtype=torch.float16):
        self.keys = list(donor_keys); self.device = device; self.dtype = dtype
        folds = [U.fold(k) for k in self.keys]
        self.stems = sorted(set.intersection(*[set(f) for f in folds]))
        self.A, self.B = {}, {}
        # CRITICAL: the operator contributes each donor's RESIDUAL, not its raw delta --
        #     M = U^T b ;  contribution = c*(b a) - c*(U M)(N V^T)
        # i.e. the donor's projection onto the shared S16 subspace is removed first. Holding raw deltas
        # here (as the first version did) amplifies the shared direction 65x on top of the S16 basis the
        # adapter already carries, and it destroyed every receiver it touched (-29pp, 0/8 improving).
        # Each donor therefore occupies 2r columns: [b, -(U M)] against [a, N V^T].
        for s in self.stems:
            Usig = U.USIG[s]
            Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
            V = U.VS[s]
            Bs, As = [], []
            for f in folds:
                b, a = f[s]                                    # b [d_out, r], a [r, d_in]
                M = Un.T @ b; N = a @ V
                Bs.append(torch.cat([b, -(Un @ M)], 1).to(dtype))   # [d_out, 2r]
                As.append(torch.cat([a, N @ V.T], 0).to(dtype))     # [2r, d_in]
            self.B[s] = torch.cat(Bs, 1).to(device)             # [d_out, K*2r]
            self.A[s] = torch.cat(As, 0).to(device)             # [K*2r, d_in]
        self.K = len(self.keys); self.r = folds[0][self.stems[0]][0].shape[1]
        print(f"[bank] {self.K} donors x {len(self.stems)} stems, rank {self.r} "
              f"-> effective rank {self.K*self.r}; {sum(t.numel() for t in self.A.values())*2/1e9:.2f}GB A + "
              f"{sum(t.numel() for t in self.B.values())*2/1e9:.2f}GB B on {device}", flush=True)

    def expand(self, w):
        """w is one coefficient per (donor, rank-1 component) = [K*r]. Each donor occupies 2r columns
        (the residual's two halves), and a component's coefficient scales BOTH halves, exactly as the
        operator's single per-donor c does."""
        if w.shape[-1] == self.K: w = w.repeat_interleave(self.r, -1)
        v = w.reshape(*w.shape[:-1], self.K, self.r)
        return torch.cat([v, v], -1).reshape(*w.shape[:-1], self.K * 2 * self.r)

    def delta(self, stem, w, h):
        """(dW_merged @ h) for one stem, in residual space. h [..., d_in]."""
        A, B = self.A[stem], self.B[stem]
        we = self.expand(w)
        z = torch.nn.functional.linear(h.to(A.dtype), A)        # [..., K*2r]
        return torch.nn.functional.linear(z * we.to(z.dtype), B)

def merged_delta_matrix(bank, stem, w):
    """Only for verification/export: materialise dW for ONE stem. Never used in the training path."""
    A, B = bank.A[stem], bank.B[stem]
    return (B * bank.expand(w).to(B.dtype)) @ A
