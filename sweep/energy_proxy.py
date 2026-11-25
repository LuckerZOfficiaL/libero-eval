#!/usr/bin/env python3
"""EXP-96 — LiST-style label-free energy as a cheap behavioural proxy for donor quality (no rollouts).

For every (target t, donor d) pair of the EXP-95 ground-truth matrix, the merged policy is exactly the probe's
operator, W = base + S16 + r_t + c * r_d (c = ref / ||r_d||), built in place on the GPU. It is run on 16 of
the TARGET's own demo frames with M = 4 flow-noise seeds (common random numbers: the same noise for every
donor of a target), batched in one sample_actions call (num_steps = 10, normalised action space). Signals:
  disp    mean variance of the M sampled action chunks          (LiST's uncertainty / stochastic-consistency)
  drift   || mean action(merged) - mean action(target alone) || (LiST's KL-to-prior)
  demo    || mean action(merged) - demonstrated action chunk ||   (imitation error on the target's own frames)
  ddisp   disp(merged) - disp(target alone)                      (energy change relative to the prior, LiST dE)
Target-alone (d = -1) is computed once per target. Output results/analysis/energy_proxy[_sN].npz:
  E[signal][t, d] with NaN where not computed. Shard over targets: argv  device  shard i/S.
"""
import json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import apply_lora as al
import unified_pool_driver as U
from reinforce_train import scenes as SC
from train_lora import load_demo_samples, _observation_from
N, K, M = 76, 16, 4
UNIQUE = [t for t in range(N) if t not in (18, 62)]
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
si, sS = map(int, (sys.argv[2] if len(sys.argv) > 2 else "0/1").split("/"))
BASE_DIR = "/media/pinas/Chris/models/pi05_base_pytorch"
OUT = REPO / f"results/analysis/energy_proxy{'' if sS == 1 else f'_s{si}'}.npz"
SIG = ["disp", "drift", "demo", "ddisp"]
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))

def main():
    t0 = time.time()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(BASE_DIR, norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model
    for p in model.parameters(): p.requires_grad_(False)
    params = dict(model.named_parameters()); sd = model.state_dict(keep_vars=True); keys = {k: None for k in sd}
    stem2p = {}
    for s_ in U.stems:
        k = al.resolve_base_key(keys, s_)
        if k is not None and k in params: stem2p[s_] = k
    pristine = {k: params[k].detach().to("cpu").pin_memory() for k in stem2p.values()}   # CPU: GPU memory is for sampling
    print(f"[ep] adapted params dtype {next(iter(params.values())).dtype}", flush=True)
    def resid_dense(key, stem, c=1.0):
        Usig = U.USIG[stem]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = U.VS[stem]
        b, a = U.fold(key)[stem]; b, a = b.to(dev), a.to(dev); Un, V = Un.to(dev), V.to(dev)
        D = b @ a; return c * (D - Un @ (Un.T @ D @ V) @ V.T)
    def add(delta_fn):
        with torch.no_grad():
            for s_, k in stem2p.items():
                d = delta_fn(s_)
                if d is None: continue
                p = params[k]
                if d.shape != p.shape: d = d.T
                p.add_(d.to(p.dtype))
    def restore(keys_=None):
        with torch.no_grad():
            for k in (keys_ or pristine):
                params[k].copy_(pristine[k], non_blocking=True)
    E = {s: np.full((N, N + 1), np.nan) for s in SIG}                         # column N = target alone
    if OUT.exists():
        z = np.load(OUT); E = {s: z[s].copy() for s in SIG}
    import os
    from subset_probe import TARGETS as WEAK
    order = [t for t in UNIQUE if t in WEAK] + [t for t in UNIQUE if t not in WEAK]     # reliable GT first
    if os.environ.get("EP_TARGETS"): mine = [int(x) for x in os.environ["EP_TARGETS"].split(",")]
    else:
        skip = {int(x) for x in os.environ.get("EP_SKIP", "").split(",") if x}
        order = [t for t in order if t not in skip]
        mine = [t for i, t in enumerate(order) if i % sS == si]
    print(f"[ep] {len(mine)} targets on {dev}; ref {REF:.3f}; setup {time.time()-t0:.0f}s", flush=True)
    for t in mine:
        if not np.isnan(E["disp"][t, [d for d in UNIQUE if d != t]]).any(): continue
        try:
            smp = load_demo_samples("libero_90_no_noops", SC[t][0], n_demos=2, stride=6)
        except FileNotFoundError:
            print(f"[ep] t{t}: no demos", flush=True); continue
        ix = np.linspace(0, len(smp) - 1, min(K, len(smp))).round().astype(int)
        obs = [smp[j][0] for j in ix]; demo = torch.tensor(np.stack([smp[j][1] for j in ix]), device=dev, dtype=torch.float32)
        prompt = SC[t][0].replace("_", " ")
        FB = int(os.environ.get("EP_FB", "4"))                                 # frames per micro-batch (x M noise)
        obs_mb = [_observation_from(policy, [o for o in obs[i:i + FB] for _ in range(M)], prompt, dev) for i in range(0, len(obs), FB)]
        g = torch.Generator(device="cpu").manual_seed(9600 + t)
        cfg = model.config; noise = torch.randn((len(obs) * M, cfg.action_horizon, cfg.action_dim), generator=g).to(dev)
        def run():
            out = []
            with torch.no_grad():
                for i, ob in enumerate(obs_mb):
                    nz = noise[i * FB * M:(i + 1) * FB * M].clone()
                    out.append(model.sample_actions(dev, ob, noise=nz, num_steps=10).float())
            a = torch.cat(out, 0)
            a = a.view(len(obs), M, *a.shape[1:])                          # [K, M, H, A]
            mean = a.mean(1); disp = a.var(1).mean().item()
            dm = (mean[..., :demo.shape[-1]] - demo).pow(2).mean().item()
            return mean, disp, dm
        # target alone in operator form: base + S16 + r_t
        # S16 + r_t computed module by module (dense dicts of both cost ~2x the adapted weights: OOM on 24 GB)
        restore(); add(lambda s_: (U.USIG[s_] @ U.VS[s_].T).to(dev) + resid_dense(t, s_) if s_ in U.fold(t) else None)
        snap = {k: params[k].detach().to("cpu").pin_memory() for k in stem2p.values()}   # target-alone weights (CPU)
        m0, d0, dm0 = run()
        E["disp"][t, N], E["drift"][t, N], E["demo"][t, N], E["ddisp"][t, N] = d0, 0.0, dm0, 0.0
        t1 = time.time()
        for d in UNIQUE:
            if d == t or not np.isnan(E["disp"][t, d]): continue
            c = REF / U.resid_norm(d)
            add(lambda s_: resid_dense(d, s_, c) if s_ in U.fold(d) else None)
            m, dsp, dm = run()
            E["disp"][t, d] = dsp; E["drift"][t, d] = (m - m0).pow(2).mean().item(); E["demo"][t, d] = dm; E["ddisp"][t, d] = dsp - d0
            with torch.no_grad():
                for k in stem2p.values(): params[k].copy_(snap[k], non_blocking=True)
                torch.cuda.synchronize(dev)
        del snap
        restore()
        np.savez(OUT, **E)
        print(f"[ep] t{t}: 73 donors in {time.time()-t1:.0f}s (target-alone disp {d0:.4f}, demo err {dm0:.4f})", flush=True)
    print(f"[ep] DONE in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
