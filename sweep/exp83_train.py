#!/usr/bin/env python3
"""EXP-83 step 6: meta-train the coefficient hypernetwork, DA-MergeLoRA style.

Each step: sample a PSEUDO-TARGET from the TRAIN split, mask it out of the donor pool (their trick), take
a batch of its own demo frames, produce coefficients from the frozen encoder's embedding of those frames,
merge the remaining donors into the frozen policy, and backprop that receiver's flow-matching loss into
the hypernetwork. The 18 held-out receivers are never seen.

RISK, stated up front and logged: the loss here is the demo action loss. EXP-22.5 showed it does not track
rollout success, and a RANDOM merge already lowers it (0.0965 -> 0.0696). A falling curve proves nothing;
only the held-out rollout evaluation does.
"""
import argparse, json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from exp83_merge import DonorBank
from exp83_hypernet import CoefHypernet, load_tokens
from exp83_patch import patch
from train_lora import load_demo_samples, BASE, _observation_from

def suite_dir(s): return s if s.endswith("_no_noops") else s + "_no_noops"

def main(a):
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    dev = a.device or f"cuda:{max(free)[1]}"
    split = json.load(open(REPO / "results/analysis/exp83_split.json"))
    # The bank is keyed by ADAPTER, but a target is an (adapter, task) pair: the goal expert is one
    # adapter serving 10 tasks, so a naive list loaded it 4x and gave it 4x the weight. Deduplicate.
    dtargets = split["donors"][:a.n_donors]
    donors, seen = [], set()
    for d in dtargets:
        if d["key"] not in seen: seen.add(d["key"]); donors.append(d["key"])
    dmeta = {d["tag"]: d for d in dtargets}
    didx = {k: i for i, k in enumerate(donors)}
    print(f"[train] {len(dtargets)} donor targets -> {len(donors)} unique adapters", flush=True)
    print(f"[train] {dev} | {len(donors)} donors | {len(split['receivers'])} held-out receivers", flush=True)

    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(BASE, norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model
    for p in model.parameters(): p.requires_grad_(False)
    bank = DonorBank(donors, device=dev, dtype=torch.bfloat16 if a.bf16 else torch.float32)
    order, ctx, _ = patch(model, bank)
    tok, band = load_tokens(order, donors, device=dev)
    stem_ix = torch.arange(len(order), device=dev)

    # query embeddings: the base policy's vision activations on each donor-task's own demo frames
    Q = torch.tensor(np.load(REPO / "results/analysis/exp83_queries.npz")["Q"], dtype=torch.float32, device=dev)
    qidx = {k: i for i, k in enumerate(json.load(open(REPO / "results/analysis/exp83_queries.json")))}
    # a donor with no demo hdf5 (f57) has no query embedding and cannot serve as a pseudo-target.
    # It STAYS in the donor pool -- it can still be merged INTO others -- it is only never sampled.
    net = CoefHypernet(d_q=(a.d_q or Q.shape[1]), n_stems=len(order), K=bank.K, r=bank.r, max_scale=a.max_scale).to(dev)
    # energy matching needs each donor's residual norm and the operator's own ref
    net.rnorm.copy_(torch.tensor([U.resid_norm(k) for k in donors], dtype=torch.float32, device=dev))
    net.ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    print(f"[dose] ref {net.ref:.3f}, donor residual norms {float(net.rnorm.min()):.2f}-{float(net.rnorm.max()):.2f}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-2)
    # EXP-83 bug 09-18: 85 identical query vectors produced a textbook donor-blind collapse that
    # looked exactly like the result we already expect. Never train on a constant input again.
    _sd = float(Q.std(0).mean())
    if _sd < 1e-3:
        raise RuntimeError(f"query embeddings do not vary across tasks (mean per-dim std {_sd:.2e}) "
                           f"-- the hypernetwork would have nothing to condition on")
    print(f"[train] query check: per-dim std across tasks {_sd:.3f}, dim {Q.shape[1]}", flush=True)
    pseudo = [d["tag"] for d in dtargets if d["tag"] in qidx]
    if len(pseudo) < len(dtargets):
        print(f"[train] {len(dtargets)-len(pseudo)} donor TARGETS lack demos -> still mergeable, never sampled", flush=True)

    cache = {}
    def batch_for(key):
        if key not in cache:
            d = dmeta[key]
            S = load_demo_samples(suite_dir(d["suite"]), d["task"], n_demos=a.n_demos, stride=a.stride)
            cache[key] = (S, d["task"].replace("_", " "))
        return cache[key]

    rng = np.random.default_rng(83); t0 = time.time(); hist = []
    for step in range(1, a.steps + 1):
        pt = pseudo[rng.integers(len(pseudo))]                  # pseudo-target (must have demos)
        S, prompt = batch_for(pt)
        ix = rng.choice(len(S), min(a.batch, len(S)), replace=False)
        obs_b = [S[i][0] for i in ix]
        act_b = torch.tensor(np.stack([S[i][1] for i in ix])).to(dev)
        observation = policy.batch_observation(obs_b, prompt) if hasattr(policy, "batch_observation") \
            else _observation_from(policy, obs_b, prompt, dev)
        w = net(Q[qidx[pt]], tok, stem_ix, band)
        mask = torch.ones(bank.K, device=dev)
        mask[didx[dmeta[pt]["key"]]] = 0.0        # mask the pseudo-target's own ADAPTER
        w = w.reshape(len(order), bank.K, bank.r) * mask[None, :, None]
        ctx[0] = w.reshape(len(order), bank.K * bank.r)
        torch.manual_seed(step)                                  # the loss is stochastic; seed per step
        loss = model(observation, act_b)
        loss = loss.mean() if torch.is_tensor(loss) and loss.ndim > 0 else loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        hist.append(float(loss))
        if step % a.log_every == 0 or step == 1:
            # COLLAPSE WATCH. EXP-30's hypernetwork died by emitting one value regardless of donor.
            # Here the analogue is the softmax concentrating on a single donor, which would show up as
            # max-share -> 1 and effective donors -> 1. Report both every log step.
            with torch.no_grad():
                W = ctx[0].reshape(len(order), bank.K, bank.r)
                share = W / W.sum(1, keepdim=True).clamp(min=1e-9)
                ent = -(share.clamp(min=1e-9).log() * share).sum(1)
                eff = float(ent.exp().mean())
                mx = float(share.max(1).values.mean())
            print(f"[train] step {step}/{a.steps} loss {np.mean(hist[-a.log_every:]):.4f} "
                  f"mass/stem {float(W.sum(1).mean()):.3f} "
                  f"max-share {mx:.3f} eff-donors {eff:.1f}/{bank.K} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save(dict(state=net.state_dict(), donors=donors, order=order, step=step,
                            hist=hist), REPO / "results/analysis/exp83_hypernet.pt")
    print(f"[train] done in {(time.time()-t0)/60:.1f} min; first/last 50 loss "
          f"{np.mean(hist[:50]):.4f} -> {np.mean(hist[-50:]):.4f}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000); p.add_argument("--batch", type=int, default=2)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--n_donors", type=int, default=68)
    p.add_argument("--n_demos", type=int, default=3); p.add_argument("--stride", type=int, default=4)
    p.add_argument("--d_q", type=int, default=0)      # 0 = take the dim from the query file
    p.add_argument("--max_scale", type=float, default=4.0); 
    p.add_argument("--log_every", type=int, default=25); p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--device", default=None)
    main(p.parse_args())
