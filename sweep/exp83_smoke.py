#!/usr/bin/env python3
"""EXP-83 step 5: end-to-end smoke test. Load the policy, patch it, push one demo batch through the
flow-matching loss, and check the gradient reaches the hypernetwork. Small donor pool for speed."""
import json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from exp83_merge import DonorBank
from exp83_hypernet import CoefHypernet, load_tokens
from exp83_patch import patch
from train_lora import load_demo_samples

def main(nd=8, dev=None):
    # GPU 0 runs EXP-81; pick the emptiest device so the two do not contend.
    if dev is None:
        import torch as _t
        free=[(_t.cuda.mem_get_info(i)[0], i) for i in range(_t.cuda.device_count())]
        dev=f"cuda:{max(free)[1]}"
        print(f"[smoke] using {dev}", flush=True)
    split = json.load(open(REPO / "results/analysis/exp83_split.json"))
    donors = [d["key"] for d in split["donors"]][:nd]
    for d in split["donors"] + split["receivers"]:
        if isinstance(d["key"], str) and d["key"].startswith("np:"): U.EXTPATH[d["key"]] = None
    t0 = time.time()
    from openpi.runtime import LiberoPolicy
    from train_lora import BASE
    policy = LiberoPolicy(BASE, norm_stats_path=str(REPO / "results/analysis/norm_assets"),
                          pytorch_device=dev)
    model = policy._model
    print(f"[smoke] policy loaded in {time.time()-t0:.0f}s", flush=True)
    for p in model.parameters(): p.requires_grad_(False)
    bank = DonorBank(donors, device=dev, dtype=torch.float32)
    order, ctx, n = patch(model, bank)
    tok, band = load_tokens(order, donors, device=dev)
    net = CoefHypernet(d_q=1152, n_stems=len(order), K=bank.K, r=bank.r).to(dev)
    print(f"[smoke] hypernet params {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    rec = split["receivers"][0]
    S = load_demo_samples(rec["suite"] if rec["suite"].endswith("_no_noops") else rec["suite"]+"_no_noops",
                          rec["task"], n_demos=1, stride=8)
    print(f"[smoke] receiver {rec['tag']} ({rec['task'][:38]}): {len(S)} demo samples", flush=True)
    obs, act = S[0]
    q = torch.randn(1152, device=dev)
    w = net(q, tok, torch.arange(len(order), device=dev), band)
    ctx[0] = w
    print(f"[smoke] coefficients {tuple(w.shape)}, injected mass/stem {w.reshape(len(order),bank.K,bank.r).sum(1).mean():.3f}", flush=True)
    # batch exactly as train_lora does, so the loss is the model's own flow-matching objective
    from train_lora import _observation_from
    B = 2
    obs_b = [S[i % len(S)][0] for i in range(B)]
    act_b = torch.tensor(np.stack([S[i % len(S)][1] for i in range(B)])).to(dev)
    prompt = rec["task"].replace("_", " ")
    observation = policy.batch_observation(obs_b, prompt) if hasattr(policy, "batch_observation") \
        else _observation_from(policy, obs_b, prompt, dev)
    loss = model(observation, act_b)
    loss = loss.mean() if torch.is_tensor(loss) and loss.ndim > 0 else loss
    print(f"[smoke] flow-matching loss = {float(loss):.4f}", flush=True)
    loss.backward()
    g = [p.grad for p in net.parameters() if p.grad is not None]
    tot = sum(float(x.norm()) for x in g)
    print(f"[smoke] GRADIENT REACHES HYPERNET: {len(g)}/{len(list(net.parameters()))} tensors, total norm {tot:.4f}", flush=True)
    # control: with zero coefficients the loss must equal the unmerged base policy's loss
    net.zero_grad(); ctx[0] = torch.zeros_like(w)
    l0 = model(observation, act_b); l0 = l0.mean() if torch.is_tensor(l0) and l0.ndim > 0 else l0
    ctx[0] = None
    lb = model(observation, act_b); lb = lb.mean() if torch.is_tensor(lb) and lb.ndim > 0 else lb
    print(f"[smoke] zero-coefficient loss {float(l0):.4f} vs unpatched base {float(lb):.4f} (must match)", flush=True)
    return dict(stems=len(order), K=bank.K, grad_norm=tot)

if __name__ == "__main__":
    print(main())
