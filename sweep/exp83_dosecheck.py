#!/usr/bin/env python3
"""EXP-83 dose check: the merged delta's Frobenius norm must match what the operator injects, and must
NOT change when the hypernetwork's shares concentrate. Over-dosing after concentration is what drove
five receivers to exactly 0% success."""
import json, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from exp83_merge import DonorBank, merged_delta_matrix
from exp83_hypernet import CoefHypernet, load_tokens

def main(nd=6):
    split = json.load(open(REPO / "results/analysis/exp83_split.json"))
    donors = [d["key"] for d in split["donors"]][:nd]
    seen, uniq = set(), []
    for k in donors:
        if k not in seen: seen.add(k); uniq.append(k)
    donors = uniq
    bank = DonorBank(donors, device="cpu", dtype=torch.float32)
    order = bank.stems
    tok, band = load_tokens(order, donors, device="cpu")
    net = CoefHypernet(d_q=32, n_stems=len(order), K=bank.K, r=bank.r)
    net.rnorm.copy_(torch.tensor([U.resid_norm(k) for k in donors], dtype=torch.float32))
    net.ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    # the operator's own 3-donor build, same stems, as ground truth
    ref = net.ref
    s0 = order[0]
    op = torch.zeros_like(bank.B[s0][:, :1] @ bank.A[s0][:1])
    for k in donors[:3]:
        c = (ref / U.resid_norm(k)) / 3
        w = torch.zeros(bank.K * bank.r); w[donors.index(k)*bank.r:(donors.index(k)+1)*bank.r] = c
        op = op + merged_delta_matrix(bank, s0, w)
    print(f"  operator (3 donors at ref/3): injected Frobenius norm {op.norm():.4f}")
    for tau, lbl in ((1.0, "spread"), (0.05, "highly concentrated")):
        net.tau = tau
        with torch.no_grad():
            w = net(torch.randn(32), tok, torch.arange(len(order)), band)
        D = merged_delta_matrix(bank, s0, w[0])
        sh = w[0].reshape(bank.K, bank.r); top = float((sh.sum(1)/sh.sum()).max())
        print(f"  hypernet, {lbl:20s} (max donor share {top:.2f}): {D.norm():.4f}  "
              f"ratio to operator {float(D.norm()/op.norm()):.2f}")
    print("  PASS if both ratios are ~1 and equal to each other (dose independent of concentration)")

if __name__ == "__main__":
    main()
