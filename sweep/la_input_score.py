#!/usr/bin/env python3
"""EXP-94 — data-aware linear algebra: how much, and in which direction, would each donor's residual change
the TARGET's layer outputs on the target's OWN inputs?  (first-order, per module, no rollouts)

For target t: apply D_t to the base policy, run 16 of its own demo frames, and at every adapted module
capture the INPUTS x. With residuals r = X Q' (operator form):
  mag[t,d] = sum_x ||r_d x||^2          = tr( X_d'X_d  . Q_d' C_t Q_d )          C_t = sum_x x x'
  dir[t,d] = sum_x <r_d x, r_t x>       = tr( X_t'X_d  . Q_d' C_t Q_t )
Accumulated per band. Scores for EXP-91-style evaluation (sweep/la_input_eval.py):
  gain  = mag / ||r_d||^2        how strongly the donor, at the operator's energy-matched dose, acts on t's data
  align = dir / sqrt(mag * mag[t,t])   cosine between the donor's and the target's own output changes on t's data
Output results/analysis/la_input_score.npz (mag, dir [5, 76, 76], rn2 [5, 76], done mask).
"""
import json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import apply_lora as al
import safetensors.torch as st
from reinforce_train import group_of, scenes as SC
from train_lora import load_demo_samples
N, NB, K = 76, 5, 16
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
BASE_DIR = "/media/pinas/Chris/models/pi05_base_pytorch"
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
OUT = REPO / "results/analysis/la_input_score.npz"

def load(t):
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(afile[t])).items():
        if g.a is None or g.b is None: continue
        r = al.infer_rank(g.a); s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]: a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = ((b * s).contiguous(), a.contiguous())
    return out

def main():
    t0 = time.time()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(BASE_DIR, norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model
    for p in model.parameters(): p.requires_grad_(False)
    _orig = model.sample_actions
    model.sample_actions = lambda device, obs, **kw: _orig(device, obs, **{"num_steps": 3, **kw})
    params = dict(model.named_parameters()); mods = dict(model.named_modules())
    sd = model.state_dict(keep_vars=True); keys = {k: None for k in sd}
    AB = [load(t) for t in range(N)]
    stems = sorted(set.intersection(*[set(x) for x in AB]))
    S = {}                                   # stem -> dict(Q, X, GXd, band, pkey)
    for s_ in stems:
        k = al.resolve_base_key(keys, s_)
        if k is None or k not in params: continue          # tied / unused (lm_head)
        mod = mods.get(k.rsplit(".weight", 1)[0])
        if not isinstance(mod, torch.nn.Linear): continue
        b = torch.stack([AB[t][s_][0] for t in range(N)]).double(); a = torch.stack([AB[t][s_][1] for t in range(N)]).double()
        dout, din = b.shape[1], a.shape[2]
        Ball = b.permute(1, 0, 2).reshape(dout, N * 16); Aall = a.reshape(N * 16, din)
        Qb, Rb = torch.linalg.qr(Ball); Qa, Ra = torch.linalg.qr(Aall.T); Us, sv, Vh = torch.linalg.svd(Rb @ Ra.T)
        U16 = Qb @ Us[:, :16]; V16 = Qa @ Vh[:16].T
        Ub = U16.T @ b; aV = a @ V16
        Bp = torch.cat([b, -(U16 @ Ub)], 2); Ap = torch.cat([a, aV @ V16.T], 1)
        Q, Ra_ = torch.linalg.qr(Ap.transpose(1, 2)); X = Bp @ Ra_.transpose(1, 2)
        Qall = Q.permute(1, 0, 2).reshape(din, N * 32); Xall = X.permute(1, 0, 2).reshape(dout, N * 32)
        S[s_] = dict(Q=Qall.to(dev, torch.float16), X=Xall.to(dev, torch.float16),
                     GXd=torch.einsum("tia,tib->tab", X, X).float().to(dev), band=group_of(s_), pkey=k, mod=mod,
                     rn2v=torch.einsum("tia,tia->t", X, X).cpu().numpy())
    print(f"[li] {len(S)} modules prepared in {(time.time()-t0)/60:.1f} min", flush=True)
    R = dict(mag=np.zeros((NB, N, N)), dir=np.zeros((NB, N, N)), rn2=np.zeros((NB, N)), done=np.zeros(N, bool))
    if OUT.exists():
        z = np.load(OUT); R = {k: z[k].copy() for k in R}
    if not R["done"].any():                                             # fresh run: residual energies per band
        for s_, v in S.items(): R["rn2"][v["band"]] += v["rn2v"]
    acc = {}; cur = {"t": None}
    def mk(s_):
        v = S[s_]
        def pre(_m, inp):
            x = inp[0]; x = x.reshape(-1, x.shape[-1]).to(torch.float16)
            Y = (x @ v["Q"]).float()                                        # [n, 76*32]
            t = cur["t"]; Yv = Y.view(-1, N, 32)
            G = torch.einsum("nda,ndb->dab", Yv, Yv)                        # Q_d' C Q_d  for every d
            Cr = Y.T @ Yv[:, t, :]                                          # Q_all' C Q_t   [76*32, 32]
            a0 = acc.setdefault(s_, [torch.zeros_like(G), torch.zeros_like(Cr)])
            a0[0] += G; a0[1] += Cr
        return pre
    handles = [v["mod"].register_forward_pre_hook(mk(s_)) for s_, v in S.items()]
    pristine = {v["pkey"]: params[v["pkey"]].detach().clone().cpu() for v in S.values()}
    for t in range(N):
        if R["done"][t]: continue
        task = SC[t][0]
        try:
            smp = load_demo_samples("libero_90_no_noops", task, n_demos=2, stride=6)
        except FileNotFoundError:
            print(f"[li] t{t}: no demos, skipped", flush=True); continue
        t1 = time.time()
        with torch.no_grad():
            for s_, v in S.items():
                b, a = AB[t][s_]; p = params[v["pkey"]]
                d = b @ a
                if d.shape != p.shape: d = d.T
                p.add_(d.to(p.dtype).to(p.device))
        cur["t"] = t; acc.clear()
        ix = np.linspace(0, len(smp) - 1, min(K, len(smp))).round().astype(int)
        rng = np.random.default_rng(t); noise = rng.standard_normal((10, 32)).astype(np.float32)
        for j in ix:
            obs, _ = smp[j]
            policy.infer({**obs, "prompt": task.replace("_", " ")}, noise=noise)
        with torch.no_grad():
            for k_, v in pristine.items(): params[k_].copy_(v.to(params[k_].device))
            for s_, (G, Cr) in acc.items():
                v = S[s_]; bd = v["band"]
                mag = (G * v["GXd"]).sum((1, 2))                                           # [76]
                Xt = v["X"][:, t * 32:(t + 1) * 32].float(); A = (Xt.T @ v["X"].float()).view(32, N, 32).permute(1, 0, 2)   # X_t'X_d [76,32,32]
                B = Cr.view(N, 32, 32)                                                     # Q_d' C Q_t
                dr = (A * B.transpose(1, 2)).sum((1, 2))                                   # tr(A B)
                R["mag"][bd, t] += mag.cpu().numpy(); R["dir"][bd, t] += dr.cpu().numpy()
        R["done"][t] = True
        np.savez(OUT, **R)
        print(f"[li] t{t} done ({len(ix)} frames, {time.time()-t1:.0f}s); {int(R['done'].sum())}/{N}", flush=True)
    for h in handles: h.remove()
    print(f"[li] DONE in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
