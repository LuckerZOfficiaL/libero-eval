#!/usr/bin/env python3
"""EXP-83 diagnostic: does the hypernetwork pick DIFFERENT donors for different receivers?

This is the check EXP-30 failed: its policy emitted the same coefficient for a target's best and worst
donor alike, i.e. it was target-conditioned but donor-blind. The analogue here is a softmax that
concentrates (good) but concentrates on the SAME donor no matter who the receiver is (degenerate).
"""
import json, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from exp83_hypernet import CoefHypernet, load_tokens
from exp83_merge import DonorBank

def main():
    ck = torch.load(REPO / "results/analysis/exp83_hypernet.pt", map_location="cpu")
    donors, order = ck["donors"], ck["order"]
    split = json.load(open(REPO / "results/analysis/exp83_split.json"))
    Q = torch.tensor(np.load(REPO / "results/analysis/exp83_queries.npz")["Q"], dtype=torch.float32)
    qidx = {k: i for i, k in enumerate(json.load(open(REPO / "results/analysis/exp83_queries.json")))}
    bank = DonorBank(donors[:2], device="cpu", dtype=torch.float32)   # tokens only need the stem list
    tok, band = load_tokens(order, donors, device="cpu")
    net = CoefHypernet(d_q=Q.shape[1], n_stems=len(order), K=len(donors), r=bank.r)
    net.load_state_dict(ck["state"]); net.eval()
    stem_ix = torch.arange(len(order))
    recv = split["receivers"]
    tops, W = [], []
    with torch.no_grad():
        for r in recv:
            w = net(Q[qidx[r["tag"]]], tok, stem_ix, band).reshape(len(order), len(donors), bank.r)
            share = (w.sum(2).sum(0) / w.sum().clamp(min=1e-9)).numpy()   # donor share, pooled over stems
            W.append(share); tops.append(int(share.argmax()))
    W = np.stack(W)
    print(f"checkpoint step {ck['step']} | {len(recv)} receivers x {len(donors)} donors")
    print(f"  distinct top-1 donors across receivers: {len(set(tops))}/{len(recv)}   tops={tops}")
    cs = W @ W.T / (np.linalg.norm(W,axis=1)[:,None]*np.linalg.norm(W,axis=1)[None,:] + 1e-9)
    off = cs[~np.eye(len(recv),dtype=bool)]
    print(f"  pairwise cosine between receivers' donor distributions: mean {off.mean():.3f} (1.0 = identical)")
    print(f"  per-donor share std across receivers: {W.std(0).mean():.4f} (0 = same mixture for everyone)")
    print("  VERDICT:", "DEGENERATE - same mixture regardless of receiver" if off.mean() > 0.99
          else "discriminating - the mixture depends on the receiver")

if __name__ == "__main__":
    main()
