#!/usr/bin/env python3
"""EXP-83 step 4: patch the frozen policy so its adapted Linears use the fused merged delta.

PEFT is bypassed on purpose: its lora_A/lora_B are leaf Parameters of rank 16, so a merge written into
them carries no gradient back to the coefficients. Instead each target Linear is wrapped:

    y = base(x) + B_all @ ( w_stem * (A_all @ x) )

with A_all/B_all frozen buffers (the donor pool) and w_stem supplied per forward pass by the hypernet.
The base weights are never modified, so one loaded policy serves every receiver.
"""
import pathlib, sys
import torch, torch.nn as nn
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))

class MergedLinear(nn.Module):
    def __init__(self, base, A, B, slot, K=None, r=None):
        super().__init__()
        self.base = base                      # frozen nn.Linear
        self.register_buffer("A", A, persistent=False)   # [K*2r, d_in]  (residual form)
        self.register_buffer("B", B, persistent=False)   # [d_out, K*2r]
        self.slot = slot                      # row of the coefficient tensor for this stem
        # Each donor occupies 2r columns (the residual's [b, -(U M)] halves) but carries only r
        # coefficients; a component's coefficient scales BOTH halves, matching the operator's single c.
        self.K, self.r = K, r
        self.ctx = None                       # set to a 1-element list holding w [S, K*r]
        self.checkpoint = True
    # The policy reads .weight/.bias off its Linears in places (dtype checks, tied paths), so a bare
    # wrapper raises AttributeError mid-forward. Forward them to the wrapped layer.
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    @property
    def in_features(self): return self.base.in_features
    @property
    def out_features(self): return self.base.out_features
    def _expand(self, w):
        if self.K is None or w.shape[-1] == self.A.shape[0]:
            return w
        v = w.reshape(*w.shape[:-1], self.K, self.r)
        return torch.cat([v, v], -1).reshape(*w.shape[:-1], self.K * 2 * self.r)

    def _delta(self, x, w):
        we = self._expand(w)
        z = torch.nn.functional.linear(x.to(self.A.dtype), self.A)
        return torch.nn.functional.linear(z * we.to(z.dtype), self.B)

    def forward(self, x, *a, **kw):
        y = self.base(x, *a, **kw)
        if self.ctx is None or self.ctx[0] is None:
            return y
        w = self.ctx[0][self.slot]
        # autograd would otherwise save z [batch, seq, K*r] for EVERY one of the 458 patched layers;
        # at K*r = 1088 that alone ran a 47GB GPU out of memory. Recompute it in backward instead --
        # it is two matmuls, far cheaper than storing it.
        if self.checkpoint and torch.is_grad_enabled():
            d = torch.utils.checkpoint.checkpoint(self._delta, x, w, use_reentrant=False)
        else:
            d = self._delta(x, w)
        return y + d.to(y.dtype)

def patch(model, bank, verbose=True):
    """Replace every adapted Linear with MergedLinear. Returns (stems_in_order, ctx, n_patched)."""
    import apply_lora as al
    sd = model.state_dict(keep_vars=True)
    mods = dict(model.named_modules())
    ctx = [None]
    order, n = [], 0
    for s in bank.stems:
        k = al.resolve_base_key({kk: None for kk in sd.keys()}, s)
        if k is None: continue
        mod_name = k.rsplit(".weight", 1)[0]
        parent_name, _, leaf = mod_name.rpartition(".")
        parent = mods.get(parent_name)
        base = mods.get(mod_name)
        if parent is None or not isinstance(base, nn.Linear): continue
        ml = MergedLinear(base, bank.A[s], bank.B[s], len(order), K=bank.K, r=bank.r)
        ml.ctx = ctx
        setattr(parent, leaf, ml)
        order.append(s); n += 1
    if verbose: print(f"[patch] wrapped {n}/{len(bank.stems)} stems", flush=True)
    return order, ctx, n

def unpatch(model):
    mods = dict(model.named_modules())
    for name, m in list(mods.items()):
        if isinstance(m, MergedLinear):
            parent_name, _, leaf = name.rpartition(".")
            setattr(mods[parent_name], leaf, m.base)
