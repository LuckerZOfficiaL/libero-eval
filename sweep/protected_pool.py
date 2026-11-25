#!/usr/bin/env python3
"""EXP-101 — protect the target, merge the WHOLE pool (no donor selection, no rollouts to choose anything).

Per target t, per adapted layer:
  C_t = covariance of the layer's INPUTS when t's own policy (base + S16 + r_t) runs on 32 of t's demo frames
  W_t = lam (C_t + lam I)^-1,  lam = ALPHA * trace(C_t) / d_in
        ~1 on input directions the target never excites, ~0 on the directions it uses
  pool delta (all 73 donors, each normalised to unit residual energy):
        unprotected  D_u = sum_d r_d / ||r_d||
        protected    D_p = sum_d (r_d W_t) / ||r_d||       -> donors only act on inputs the target does not see
Arms (merged into W = base + S16 + r_t + s * D):
  u1 : unprotected pool, total ||s D|| = ref (one donor's operator dose)       -- control
  p1 : protected pool, total = ref
  p3 : protected pool, total = 3 ref                                            -- does protection allow more?
Each arm is written as a full checkpoint in /dev/shm and served directly (orchestrator 'prebuilt_dir').
Diagnostics per target: energy retained by protection, and the first-order change of t's layer outputs on its
own states, tr(D C_t D'), protected vs unprotected.  Records e101_t{t}_{arm}.
Env: PP_TARGETS (default: 15 weak), PP_ALPHA (default 0.01), PP_ARMS (u1,p1,p3), PP_ROUND (targets per round, 2).
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
import apply_lora as al
import unified_pool_driver as U
import la_pair_features as LF
from reinforce_train import scenes as SC
from train_lora import load_demo_samples
from subset_probe import TARGETS as WEAK

N = 76; UNIQUE = [t for t in range(N) if t not in (18, 62)]
BASE_DIR = pathlib.Path("/media/pinas/Chris/models/pi05_base_pytorch")
PRE = pathlib.Path("/dev/shm/mergesweep/prebuilt")
ALPHA = float(os.environ.get("PP_ALPHA", "0.01"))
ARMS = os.environ.get("PP_ARMS", "u1,p1,p3").split(",")
SCALE = {"u1": 1.0, "p1": 1.0, "p3": 3.0}
TGTS = [int(x) for x in os.environ["PP_TARGETS"].split(",")] if os.environ.get("PP_TARGETS") else list(WEAK)
ROUND = int(os.environ.get("PP_ROUND", "2"))
OFFLOAD = os.environ.get("PP_OFFLOAD") == "1"                                   # odin: driver shares a GPU with servers
LOWMEM = os.environ.get("PP_LOWMEM") == "1"                                     # odin (62GB RAM): bf16 pool deltas, D_t rebuilt at write
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
OUT_DIAG = REPO / "results/analysis/exp101_diag.json"

def main():
    t0 = time.time()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(str(BASE_DIR), norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    params = dict(model.named_parameters()); mods = dict(model.named_modules())
    keys = {k: None for k in model.state_dict(keep_vars=True)}
    AB = [LF.load(d) for d in range(N)]
    stems = [s for s in U.stems if all(s in AB[d] for d in range(N))]
    S = {}
    for s_ in stems:
        k = al.resolve_base_key(keys, s_)
        if k is None or k not in params: continue
        m = mods.get(k.rsplit(".weight", 1)[0])
        if not isinstance(m, torch.nn.Linear): continue
        S[s_] = dict(k=k, mod=m)
    base_sd = st.load_file(str(BASE_DIR / "model.safetensors"))                 # CPU, bf16 -- written into every arm
    ck_key = {s_: al.resolve_base_key({k: None for k in base_sd}, s_) for s_ in S}
    if LOWMEM:                                                                 # reuse the file's tensors, no second copy
        pristine = {v["k"]: base_sd[ck_key[s_]] for s_, v in S.items()}
        assert all(pristine[v["k"]].shape == params[v["k"]].shape for v in S.values())
        for k_ in list(pristine)[:5]:                                         # file tensors == loaded weights
            assert torch.equal(pristine[k_].to(params[k_].dtype), params[k_].detach().cpu()), k_
    else:
        pristine = {v["k"]: params[v["k"]].detach().to("cpu").pin_memory() for v in S.values()}
    print(f"[pp] {len(S)} modules; alpha {ALPHA}; ref {REF:.3f}; setup {(time.time()-t0)/60:.1f} min", flush=True)

    def factors(d, s_):
        Usig = U.USIG[s_]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = U.VS[s_]
        b, a = AB[d][s_]; M = Un.T @ b; Nn = a @ V
        return torch.cat([b, -(Un @ M)], 1), torch.cat([a, Nn @ V.T], 0)                    # r = Bp @ Ap

    rnorm = np.array([U.resid_norm(d) for d in range(N)])
    diag = json.load(open(OUT_DIAG)) if OUT_DIAG.exists() else {}
    todo = [t for t in TGTS if not all(U.get(f"e101_t{t}_{a}", SC[t][0]) is not None for a in ARMS)]
    for r0 in range(0, len(todo), ROUND):
        jobs = []
        for t in todo[r0:r0 + ROUND]:
            t1 = time.time()
            donors = [d for d in UNIQUE if d != t]
            # ---- 1. target's operator weights in place, input Grams over its demo frames
            Dt = {}
            with torch.no_grad():
                for k_, v in pristine.items(): params[k_].copy_(v.to(dev))
                for s_, v in S.items():
                    Bt, At = factors(t, s_)
                    D = (U.USIG[s_] @ U.VS[s_].T + Bt @ At)
                    if not LOWMEM: Dt[s_] = D
                    p = params[v["k"]]; p.add_((D if D.shape == p.shape else D.T).to(dev, p.dtype))
            smp = load_demo_samples("libero_90_no_noops", SC[t][0], n_demos=4, stride=4)
            ix = np.linspace(0, len(smp) - 1, min(32, len(smp))).round().astype(int)
            names = list(S); half = len(names) // 2
            G, ntok = {}, {}
            for group in (names[:half], names[half:]):                          # two passes: Grams for half the layers each
                hooks = []
                for s_ in group:
                    def mk(s_):
                        def pre(_m, inp):
                            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                            G[s_] = G.get(s_, 0) + x.T @ x; ntok[s_] = ntok.get(s_, 0) + x.shape[0]
                        return pre
                    hooks.append(S[s_]["mod"].register_forward_pre_hook(mk(s_)))
                for j in ix:
                    obs, _ = smp[j]; policy.infer({**obs, "prompt": SC[t][0].replace("_", " ")})
                for h in hooks: h.remove()
                # ---- 2. per layer: protection operator and the two pool deltas
                for s_ in group:
                    if s_ not in G:                                                # never called at inference (expert lm_head)
                        Bs, As = zip(*[factors(d, s_) for d in donors])
                        Du = torch.cat([b / rnorm[d] for b, d in zip(Bs, donors)], 1) @ torch.cat(As, 0)
                        if LOWMEM: Du = Du.to(torch.bfloat16)
                        S[s_]["Du"], S[s_]["Dp"], S[s_]["eu"], S[s_]["ep"] = Du, Du.clone(), 0.0, 0.0
                        continue
                    C = G.pop(s_) / ntok[s_]; din = C.shape[0]
                    lam = max(ALPHA * float(torch.trace(C)) / din, 1e-8)
                    Lc = torch.linalg.cholesky(C + lam * torch.eye(din, device=dev))
                    Bs, As = zip(*[factors(d, s_) for d in donors])
                    Bk = torch.cat([b / rnorm[d] for b, d in zip(Bs, donors)], 1).to(dev)          # unit-energy donors
                    Aa = torch.cat(As, 0).to(dev)                                                  # [73*32, d_in]
                    Wt = lam * torch.cholesky_solve(Aa.T.contiguous(), Lc)                         # W_t Ap^T  [d_in, 73*32]
                    Du = Bk @ Aa; Dp = Bk @ Wt.T
                    S[s_]["eu"] = float((Du @ C * Du).sum()); S[s_]["ep"] = float((Dp @ C * Dp).sum())   # tr(D C D')
                    S[s_]["nu2"], S[s_]["np2"] = float((Du ** 2).sum()), float((Dp ** 2).sum())
                    sto = torch.bfloat16 if LOWMEM else torch.float32
                    S[s_]["Du"], S[s_]["Dp"] = Du.to(sto).cpu(), Dp.to(sto).cpu()
                    del C, Lc, Wt, Du, Dp, Aa, Bk
                torch.cuda.empty_cache()
            nu = sum(v.get("nu2", float((v["Du"].float() ** 2).sum())) for v in S.values()) ** 0.5
            npr = sum(v.get("np2", float((v["Dp"].float() ** 2).sum())) for v in S.values()) ** 0.5
            eu = sum(v["eu"] for v in S.values()); ep = sum(v["ep"] for v in S.values())
            # first-order output change on the target's own states, at EQUAL injected norm (ref)
            diag[str(t)] = dict(energy_retained=npr / nu, out_change_unprot=eu * (REF / nu) ** 2,
                                out_change_prot=ep * (REF / npr) ** 2, ratio=(ep / npr ** 2) / (eu / nu ** 2))
            json.dump(diag, open(OUT_DIAG, "w"), indent=1)
            print(f"[pp] t{t}: protection keeps {100*npr/nu:.1f}% of pool energy; output change on target states at equal norm: "
                  f"protected/unprotected = {diag[str(t)]['ratio']:.4f}  ({(time.time()-t1)/60:.1f} min)", flush=True)
            # ---- 3. write one checkpoint per arm: base + (S16 + r_t) + s * D
            for arm in ARMS:
                if U.get(f"e101_t{t}_{arm}", SC[t][0]) is not None: continue
                s = SCALE[arm] * REF / (nu if arm.startswith("u") else npr)
                out = PRE / f"pp_{t}_{arm}"; shutil.rmtree(out, ignore_errors=True); (out / "assets/physical-intelligence/libero").mkdir(parents=True)
                sd = dict(base_sd)
                for s_, v in S.items():
                    ck = ck_key[s_]; W = base_sd[ck].float()
                    if LOWMEM: Bt, At = factors(t, s_); D_t = U.USIG[s_] @ U.VS[s_].T + Bt @ At
                    else: D_t = Dt[s_]
                    D = D_t + s * (v["Du"] if arm.startswith("u") else v["Dp"]).float()
                    sd[ck] = (W + (D if D.shape == W.shape else D.T)).to(base_sd[ck].dtype)
                st.save_file(sd, str(out / "model.safetensors"))
                shutil.copy2(BASE_DIR / "config.json", out / "config.json")
                shutil.copy2(REPO / "results/analysis/norm_assets/norm_stats.json", out / "assets/physical-intelligence/libero/norm_stats.json")
                jobs.append({"name": f"e101_t{t}_{arm}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [],
                             "prebuilt_dir": str(out), "eval_scenes": [SC[t][0]], "suite": "libero_90", "norm_stats": U.FLEET_NORM})
            for v in S.values(): [v.pop(k_, None) for k_ in ("Du", "Dp", "nu2", "np2")]
            del Dt
        # ---- 4. evaluate this round's checkpoints (25 rollouts each), then free /dev/shm
        if jobs:
            with torch.no_grad():
                for k_, v in pristine.items(): params[k_].copy_(v.to(dev))
            planp = REPO / f"sweep/_pp_{os.getpid()}.json"
            json.dump(dict(U.META, n_trials=U.COMMIT_R, jobs=jobs), open(planp, "w"), indent=1)
            if OFFLOAD: model.to("cpu"); torch.cuda.empty_cache()                       # free the GPU for the servers
            subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)
            if OFFLOAD: model.to(dev)
            for j in jobs: shutil.rmtree(j["prebuilt_dir"], ignore_errors=True)
        print(f"[pp] round {r0//ROUND+1}/{(len(todo)+ROUND-1)//ROUND} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    print("[pp] DONE", flush=True)

if __name__ == "__main__":
    main()
