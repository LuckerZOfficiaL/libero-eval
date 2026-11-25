#!/usr/bin/env python3
"""EXP-99 — gradient-based compatibility (Demystifying Mergeability's most stable predictors) + Fisher energy.

For every task t (74), at its operator point W_t = base + S16 + r_t, the policy's own training loss (flow
matching, fixed noise/time per sample) on 16 of t's demo frames, one backward per frame. Hooks on the adapted
weights reduce each per-sample gradient G immediately (never stored whole):
  proj[i, d] = <G_i, r_d>            for all 76 donors (r_d = operator-form residual, factored Bp_d Ap_d)
  sketch_i   = R' G_i C              per module, R (d_out x 8), C (d_in x 8) fixed Gaussians -> 64 numbers
Per (t, d), per band:
  lookahead = c_d * mean_i proj[i, d]          first-order change of t's loss if d is added (TAG analogue)
  fisher    = c_d^2 * mean_i proj[i, d]^2      empirical-Fisher energy of the donor step on t's data
  gcos, gl2 = cosine / L2 between t's and d's mean-gradient sketches (each at its own point, own data)
Pre-registered: lower lookahead (donor goes downhill for t), lower fisher, higher gcos, lower gl2 = better.
Output results/analysis/grad_proxy[_sN].npz.  argv: device  shard i/S
"""
import os, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import apply_lora as al
import unified_pool_driver as U
from reinforce_train import scenes as SC, group_of
from train_lora import load_demo_samples, _observation_from
N, K, NB, SK = 76, 16, 5, 8
UNIQUE = [t for t in range(N) if t not in (18, 62)]
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
si, sS = map(int, (sys.argv[2] if len(sys.argv) > 2 else "0/1").split("/"))
OUT = REPO / f"results/analysis/grad_proxy{'' if sS == 1 else f'_s{si}'}.npz"
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))

def main():
    t0 = time.time()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy("/media/pinas/Chris/models/pi05_base_pytorch", norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model
    for p in model.parameters(): p.requires_grad_(False)
    model.eval()                                                            # deterministic loss (no dropout)
    params = dict(model.named_parameters()); sd = model.state_dict(keep_vars=True); keys = {k: None for k in sd}
    S = {}
    gsk = torch.Generator().manual_seed(99)
    import la_pair_features as LF                                            # load each adapter ONCE (U.fold's
    AB = [LF.load(d) for d in range(N)]                                      # 14-entry cache thrashed: 27 min setup)
    for s_ in U.stems:
        k = al.resolve_base_key(keys, s_)
        if k is None or k not in params: continue
        Usig = U.USIG[s_]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = U.VS[s_]
        Bs, As = [], []
        if not all(s_ in AB[d] for d in range(N)): continue
        for d in range(N):
            b, a = AB[d][s_]; M = Un.T @ b; Nn = a @ V
            Bs.append(torch.cat([b, -(Un @ M)], 1)); As.append(torch.cat([a, Nn @ V.T], 0))
        dout, din = Bs[0].shape[0], As[0].shape[1]
        S[s_] = dict(k=k, band=group_of(s_), B=torch.cat(Bs, 1).to(dev, torch.float16), A=torch.cat(As, 0).to(dev, torch.float16),
                     R=torch.randn(dout, SK, generator=gsk).to(dev) / np.sqrt(SK), C=torch.randn(din, SK, generator=gsk).to(dev) / np.sqrt(SK),
                     shape=(dout, din))
    print(f"[gp] {len(S)} modules prepared in {(time.time()-t0)/60:.1f} min", flush=True)
    k2s = {v["k"]: s_ for s_, v in S.items()}
    cur = {}; KEEP = {"v": bool(os.environ.get("GP_VALIDATE"))}
    def mk(s_):
        v = S[s_]
        def hook(p):
            G = p.grad.float()
            if G.shape != v["shape"]: G = G.T
            # fp32 throughout: gradients are ~1e-6 and underflow in fp16 (first validation: hook 100x too small)
            Mx = G @ v["A"].float().T                                       # [d_out, N*32]
            proj = (v["B"].float() * Mx).sum(0).view(N, 32).sum(1)          # <G, r_d> for all d
            sk = (v["R"].T @ G @ v["C"]).flatten()
            cur.setdefault("proj", torch.zeros(NB, N, device=dev))[v["band"]] += proj
            cur.setdefault("sk", {})[s_] = sk
            if not KEEP["v"]: p.grad = None
        return hook
    for s_, v in S.items():
        params[v["k"]].requires_grad_(True); params[v["k"]].register_post_accumulate_grad_hook(mk(s_))
    pristine = {v["k"]: params[v["k"]].detach().to("cpu").pin_memory() for v in S.values()}
    names = sorted(S)
    R_ = dict(proj=np.full((N, K, NB, N), np.nan), sketch=np.full((N, len(names) * SK * SK), np.nan, dtype=np.float32),
              band_of=np.array([S[n]["band"] for n in names]), done=np.zeros(N, bool))
    if OUT.exists():
        z = np.load(OUT); R_ = {k: z[k].copy() for k in R_}
    from subset_probe import TARGETS as WEAK
    order = [t for t in UNIQUE if t in WEAK] + [t for t in UNIQUE if t not in WEAK]
    mine = [t for i, t in enumerate(order) if i % sS == si]
    for t in mine:
        if R_["done"][t]: continue
        try:
            smp = load_demo_samples("libero_90_no_noops", SC[t][0], n_demos=2, stride=6)
        except FileNotFoundError:
            print(f"[gp] t{t}: no demos", flush=True); continue
        t1 = time.time()
        with torch.no_grad():                                               # operator point: base + S16 + r_t
            for k_, v in pristine.items(): params[k_].copy_(v.to(dev))
            for s_, v in S.items():
                if s_ not in U.fold(t): continue
                Usig = U.USIG[s_].to(dev); V = U.VS[s_].to(dev)
                D = Usig @ V.T + v["B"][:, t * 32:(t + 1) * 32].float() @ v["A"][t * 32:(t + 1) * 32].float()
                p = params[v["k"]]; p.add_((D if D.shape == p.shape else D.T).to(p.dtype))
        ix = np.linspace(0, len(smp) - 1, min(K, len(smp))).round().astype(int)
        prompt = SC[t][0].replace("_", " ")
        g = torch.Generator().manual_seed(9900 + t)
        sk_sum = None
        for i, j in enumerate(ix):
            obs, act = smp[j]
            ob = _observation_from(policy, [obs], prompt, dev); at = torch.tensor(act[None], device=dev)
            nz = torch.randn(at.shape, generator=g).to(dev); tm = (torch.rand(1, generator=g) * 0.999 + 0.001).to(dev)
            cur.clear()
            loss = model(ob, at, noise=nz, time=tm).mean(); loss.backward()
            R_["proj"][t, i] = cur["proj"].cpu().numpy()
            if os.environ.get("GP_VALIDATE") and i == 0:                       # validation, then exit
                for dchk in (19, 50):
                    direct = 0.0
                    for s_, v in S.items():                                     # explicit <full grad, r_d>
                        p = params[v["k"]]
                        if p.grad is None: continue
                        Dd = v["B"][:, dchk * 32:(dchk + 1) * 32].float() @ v["A"][dchk * 32:(dchk + 1) * 32].float()
                        G = p.grad.float(); G = G if G.shape == Dd.shape else G.T
                        direct += float((G * Dd).sum())
                    line = f"[gp] VALIDATE t{t} d{dchk}: hook {float(cur['proj'][:, dchk].sum()):.5e}  direct {direct:.5e} |"
                    for eps in (0.05, 0.2):
                        for rep in range(2):
                            fd = []
                            for sgn in (1, -1):
                                with torch.no_grad():
                                    for s_, v in S.items():
                                        Dd = v["B"][:, dchk * 32:(dchk + 1) * 32].float() @ v["A"][dchk * 32:(dchk + 1) * 32].float()
                                        p = params[v["k"]]; p.add_(sgn * eps * (Dd if Dd.shape == p.shape else Dd.T))
                                    fd.append(float(model(ob, at, noise=nz, time=tm).mean()))
                                    for s_, v in S.items():
                                        Dd = v["B"][:, dchk * 32:(dchk + 1) * 32].float() @ v["A"][dchk * 32:(dchk + 1) * 32].float()
                                        p = params[v["k"]]; p.add_(-sgn * eps * (Dd if Dd.shape == p.shape else Dd.T))
                            line += f" fd(eps={eps},rep{rep}) {(fd[0]-fd[1])/(2*eps):.5e}"
                    print(line, flush=True)
                with torch.no_grad():
                    l0 = [float(model(ob, at, noise=nz, time=tm).mean()) for _ in range(3)]
                print(f"[gp] VALIDATE repeat loss at the same point: {l0}", flush=True)
                return
            z64 = torch.zeros(SK * SK, device=dev)                             # modules with no gradient (unused lm_head)
            v_ = torch.cat([cur["sk"].get(n, z64) for n in names]).cpu().numpy()
            sk_sum = v_ if sk_sum is None else sk_sum + v_
        R_["sketch"][t] = sk_sum / len(ix); R_["done"][t] = True
        np.savez(OUT, **R_)
        print(f"[gp] t{t} done in {time.time()-t1:.0f}s; {int(R_['done'].sum())} tasks in this shard file", flush=True)
    print(f"[gp] DONE in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
