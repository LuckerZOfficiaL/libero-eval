#!/usr/bin/env python3
"""EXP-83 control: with the flow-matching noise/time draw FIXED, zero coefficients must reproduce the
unpatched base loss exactly, and a non-zero merge must change it. Without seeding, the loss is stochastic
and the comparison is meaningless (that is what made the first control look like a failure)."""
import json, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from exp83_merge import DonorBank
from exp83_hypernet import CoefHypernet, load_tokens
from exp83_patch import patch, unpatch
from train_lora import load_demo_samples, BASE, _observation_from

def loss_at(model, observation, act_b, seed=7):
    torch.manual_seed(seed); np.random.seed(seed)
    l = model(observation, act_b)
    return float(l.mean() if torch.is_tensor(l) and l.ndim > 0 else l)

def main(nd=8):
    free=[(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    dev=f"cuda:{max(free)[1]}"; print(f"[ctl] {dev}", flush=True)
    split = json.load(open(REPO / "results/analysis/exp83_split.json"))
    donors = [d["key"] for d in split["donors"]][:nd]
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(BASE, norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model
    for p in model.parameters(): p.requires_grad_(False)
    rec = split["receivers"][0]
    S = load_demo_samples(rec["suite"]+"_no_noops" if not rec["suite"].endswith("_no_noops") else rec["suite"],
                          rec["task"], n_demos=1, stride=8)
    obs_b = [S[i % len(S)][0] for i in range(2)]
    act_b = torch.tensor(np.stack([S[i % len(S)][1] for i in range(2)])).to(dev)
    prompt = rec["task"].replace("_", " ")
    observation = policy.batch_observation(obs_b, prompt) if hasattr(policy, "batch_observation") \
        else _observation_from(policy, obs_b, prompt, dev)
    base_a = loss_at(model, observation, act_b); base_b = loss_at(model, observation, act_b)
    print(f"[ctl] unpatched, same seed twice: {base_a:.6f} / {base_b:.6f}  -> deterministic: {abs(base_a-base_b)<1e-9}", flush=True)
    bank = DonorBank(donors, device=dev, dtype=torch.float32)
    order, ctx, n = patch(model, bank)
    tok, band = load_tokens(order, donors, device=dev)
    net = CoefHypernet(d_q=1152, n_stems=len(order), K=bank.K, r=bank.r).to(dev)
    ctx[0] = torch.zeros(len(order), bank.K*bank.r, device=dev)
    zero = loss_at(model, observation, act_b)
    ctx[0] = net(torch.randn(1152, device=dev), tok, torch.arange(len(order), device=dev), band)
    merged = loss_at(model, observation, act_b)
    print(f"[ctl] ZERO coefficients : {zero:.6f}   (base {base_a:.6f}, diff {abs(zero-base_a):.2e})  PASS={abs(zero-base_a)<1e-6}", flush=True)
    print(f"[ctl] MERGED (mass 0.5) : {merged:.6f}   (differs from base by {merged-base_a:+.4f})  PASS={abs(merged-base_a)>1e-4}", flush=True)
    unpatch(model)
    after = loss_at(model, observation, act_b)
    print(f"[ctl] after unpatch     : {after:.6f}   PASS={abs(after-base_a)<1e-6}", flush=True)

if __name__ == "__main__":
    main()
