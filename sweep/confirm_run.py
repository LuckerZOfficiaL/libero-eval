#!/usr/bin/env python3
"""EXP-60 (rewritten) — pre-registered confirmation of dnorm_llm_17 on a NEW target cohort.

Stage 1 of the first attempt named jobs with a 28-character prefix of the adapter directory, and the
LIBERO-90 variant directories share their first 28 characters -- so ten targets collapsed into three job
names and only 24 of 80 probes were real. Job names here use a stable md5 id instead. The signature path
also assumed integer fleet targets (reinforce_train.fold indexes a fleet table), which these S16-init
variants are not, so this uses unified_pool_driver.fold, which accepts both.

PRE-REGISTERED, unchanged from before the first attempt:
  feature dnorm_llm_17, positive sign; 10 weak S16-init variants x the same 8 fixed donors;
  prediction interaction ~ +0.35 and top-3 worth ~40% of the oracle gap; near zero refutes it.
"""
import argparse, hashlib, json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import os; os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import unified_pool_driver as U
h6 = lambda n: hashlib.md5(n.encode()).hexdigest()[:8]
DES = json.load(open(REPO / "results/analysis/shared_donor_design.json")); DONORS = DES["donors"]
TG = json.load(open(REPO / "results/analysis/confirm_design.json"))
for g in TG:
    U.EXTPATH["np:" + g["name"]] = g["adapter"]
HOOKS = {"vision": "paligemma_with_expert.paligemma.model.vision_tower",
         "llm_00": "paligemma_with_expert.paligemma.model.language_model.layers.0",
         "llm_05": "paligemma_with_expert.paligemma.model.language_model.layers.5",
         "llm_11": "paligemma_with_expert.paligemma.model.language_model.layers.11",
         "llm_17": "paligemma_with_expert.paligemma.model.language_model.layers.17",
         "exp_00": "paligemma_with_expert.gemma_expert.model.layers.0",
         "exp_08": "paligemma_with_expert.gemma_expert.model.layers.8",
         "exp_17": "paligemma_with_expert.gemma_expert.model.layers.17",
         "act_out": "action_out_proj"}
OUT = REPO / "results/analysis/confirm_sigs"; OUT.mkdir(parents=True, exist_ok=True)

def probes():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(g, d) for g in TG for d in DONORS if U.get(f"e65_{h6(g['name'])}_p{d}", g["task"]) is None]
    print(f"[cg2] {len(todo)} probes at 10 rollouts", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for g, d in todo[i:i + U.CHUNK]:
            p = U.build(f"cg2_{h6(g['name'])}_{d}", "np:" + g["name"], [(d, ref / U.resid_norm(d))]); built.append(p)
            jobs.append({"name": f"e65_{h6(g['name'])}_p{d}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [g["task"]], "_suite": g["suite"]})
        for su in sorted({j["_suite"] for j in jobs}):
            U.run([{k: v for k, v in j.items() if k != "_suite"} for j in jobs if j["_suite"] == su], 10, su, U.FLEET_NORM)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[cg2] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[cg2] probes done", flush=True)

def sigs(device):
    from reinforce_train import stems, USIG, VS
    import apply_lora as al
    from openpi.runtime import LiberoPolicy
    need = [(g, None) for g in TG if not (OUT / f"{h6(g['name'])}_base.npz").exists()]
    need += [(g, d) for g in TG for d in DONORS if not (OUT / f"{h6(g['name'])}_{d}.npz").exists()]
    print(f"[cg2] {len(need)} signatures to compute", flush=True)
    if not need: return
    policy = LiberoPolicy("/media/pinas/Chris/models/pi05_base_pytorch",
                          norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=device)
    model = policy._model; params = dict(model.named_parameters()); sd = model.state_dict()
    s2p = {}
    for s_ in stems:
        k = al.resolve_base_key({k: None for k in sd.keys()}, s_)
        if k is not None and k in params: s2p[s_] = k
    pristine = {k: params[k].detach().clone().cpu() for k in set(s2p.values())}
    def first_tensor(x):
        if torch.is_tensor(x): return x
        if isinstance(x, (tuple, list)):
            for e in x:
                t_ = first_tensor(e)
                if t_ is not None: return t_
        return getattr(x, "last_hidden_state", None)
    acts, handles = {}, []
    for tag, mn in HOOKS.items():
        def mk(tag):
            def hook(_m, _i, out):
                t_ = first_tensor(out)
                if t_ is not None: acts[tag] = t_.float().mean(dim=tuple(range(t_.ndim - 1))).detach().cpu()
            return hook
        handles.append(dict(model.named_modules())[mn].register_forward_hook(mk(tag)))
    noise = np.random.default_rng(42).standard_normal((10, 32)).astype(np.float32)
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    for n, (g, d) in enumerate(need):
        t0 = time.time(); key = "np:" + g["name"]
        ft = U.fold(key); fd = U.fold(d) if d is not None else {}
        c = (ref / U.resid_norm(d)) if d is not None else 0.0
        with torch.no_grad():
            for s_ in stems:
                if s_ not in s2p: continue
                Usig = USIG[s_]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = VS[s_]
                M = Usig @ V.T
                for f, k in ((ft, 1.0), (fd, c)):
                    if s_ in f and k:
                        b, a = f[s_]; D = b @ a
                        M = M + k * (D - Un @ (Un.T @ D @ V) @ V.T)
                p = params[s2p[s_]]
                p.add_((M if tuple(p.shape) == tuple(M.shape) else M.T).to(p.dtype).to(p.device))
        bank = np.load(REPO / f"results/analysis/obs_banks/{g['name']}.npz")
        sig = {tag: [] for tag in HOOKS}
        for i in range(len(bank["state"])):
            policy.infer({"observation/image": bank["img"][i], "observation/wrist_image": bank["wrist"][i],
                          "observation/state": bank["state"][i], "prompt": str(bank["prompt"][i])}, noise=noise)
            miss = [tg for tg in HOOKS if tg not in acts]
            if miss: raise RuntimeError(f"no activation at {miss}")
            for tag in HOOKS: sig[tag].append(acts[tag].numpy().astype(np.float16))
        with torch.no_grad():
            for k, v in pristine.items(): params[k].copy_(v.to(params[k].device))
        np.savez_compressed(OUT / (f"{h6(g['name'])}_base.npz" if d is None else f"{h6(g['name'])}_{d}.npz"),
                            **{k: np.stack(v) for k, v in sig.items()})
        print(f"[cg2] {n+1}/{len(need)} {h6(g['name'])}_{d}  {time.time()-t0:.0f}s", flush=True)
    for h in handles: h.remove()
    print("[cg2] signatures done", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="probes", choices=["probes", "sigs"])
    ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    probes() if a.stage == "probes" else sigs(a.device)
