#!/usr/bin/env python3
"""ROUTE 1 — interrogate the MERGED model, rollout-free.

Every rollout-free selector we built was a function of (target features, donor features) and failed.
This computes something we never looked at: what the MERGE ITSELF does to the policy's behaviour.

For each (target, donor) pair, build the committed operator in memory
    W = S16 + r_target + c * r_donor        (c = energy-matched, exactly as the probe builds it)
apply it to the base model, run the fixed 180-observation bank, and record activations at 9 depths.
No environment, no episodes. The reference is the target alone -- and since S16 + r_target = D_target
exactly, that signature already exists in act_sigs/task<t>.npz, so only the merged ones are new.

The question: does "how far, and where, this merge moved the policy" predict the probe's verdict?
Labels are the 10-rollout probe scores we already own (e27), so this needs ZERO new rollouts.
"""
import argparse, json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import os; os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import safetensors.torch as st, apply_lora as al
from reinforce_train import fold, stems, USIG, VS, target_baseline

HOOKS = {"vision": "paligemma_with_expert.paligemma.model.vision_tower",
         "llm_00": "paligemma_with_expert.paligemma.model.language_model.layers.0",
         "llm_05": "paligemma_with_expert.paligemma.model.language_model.layers.5",
         "llm_11": "paligemma_with_expert.paligemma.model.language_model.layers.11",
         "llm_17": "paligemma_with_expert.paligemma.model.language_model.layers.17",
         "exp_00": "paligemma_with_expert.gemma_expert.model.layers.0",
         "exp_08": "paligemma_with_expert.gemma_expert.model.layers.8",
         "exp_17": "paligemma_with_expert.gemma_expert.model.layers.17",
         "act_out": "action_out_proj"}
BASE = "/media/pinas/Chris/models/pi05_base_pytorch"


def merged_delta(t, d, c):
    """W - W_base for the committed operator, per stem, in float32."""
    ft = fold(t); fd = fold(d) if d is not None else {}
    out = {}
    for s_ in stems:
        Usig = USIG[s_]; U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = VS[s_]
        M = Usig @ V.T                                   # S16
        if s_ in ft:
            b, a = ft[s_]; D = b @ a
            M = M + (D - U @ (U.T @ D @ V) @ V.T)        # + r_target
        if d is not None and s_ in fd:
            b, a = fd[s_]; D = b @ a
            M = M + c * (D - U @ (U.T @ D @ V) @ V.T)    # + c * r_donor
        out[s_] = M
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="weak", help="'weak' = the 15 fleet weak targets, or comma list")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bank", default="shared",
                    help="shared = the 180-obs cross-scene bank; target = each target's own demo states "
                         "(results/analysis/obs_banks/t<T>.npz), which is what actually determines whether "
                         "a merge helps THAT task")
    a = ap.parse_args()
    RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
    tg = ([t for t in range(76) if str(t) in RANK and target_baseline(t) < 0.9] if a.targets == "weak"
          else [int(x) for x in a.targets.split(",")])
    pairs = globals().get("PAIR_OVERRIDE") or [(t, int(p)) for t in tg for p, _ in RANK[str(t)]]
    outdir = REPO / ("results/analysis/merged_sigs" if a.bank == "shared" else "results/analysis/merged_sigs_tb")
    outdir.mkdir(parents=True, exist_ok=True)
    todo = [(t, d) for t, d in pairs if not (outdir / f"m{t}_{d}.npz").exists()]
    if a.bank == "target":   # the target alone on its own bank is the reference signature
        todo = [(t, None) for t in sorted({t for t, _ in pairs}) if not (outdir / f"m{t}_base.npz").exists()] + todo
    print(f"[msig] {len(tg)} targets, {len(pairs)} pairs, {len(todo)} to compute", flush=True)
    if not todo: return

    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(BASE, norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=a.device)
    model = policy._model
    params = dict(model.named_parameters())
    sd = model.state_dict()
    stem2param = {}
    for s_ in stems:
        k = al.resolve_base_key({k: None for k in sd.keys()}, s_)
        if k is not None and k in params: stem2param[s_] = k
    pristine = {k: params[k].detach().clone().cpu() for k in set(stem2param.values())}
    print(f"[msig] {len(stem2param)} modules; pristine {sum(v.numel()*v.element_size() for v in pristine.values())/1e9:.1f} GB", flush=True)
    def first_tensor(x):
        """Hooked modules return tensors, tuples, or HF ModelOutput objects; the vision tower returns
        the last (hence the earlier KeyError when only tensors/tuples were handled)."""
        if torch.is_tensor(x): return x
        if isinstance(x, (tuple, list)):
            for e in x:
                t_ = first_tensor(e)
                if t_ is not None: return t_
        if hasattr(x, "last_hidden_state"): return x.last_hidden_state
        return None

    acts, handles = {}, []
    for tag, mn in HOOKS.items():
        def mk(tag):
            def hook(_m, _i, out):
                t_ = first_tensor(out)
                if t_ is not None: acts[tag] = t_.float().mean(dim=tuple(range(t_.ndim - 1))).detach().cpu()
            return hook
        handles.append(dict(model.named_modules())[mn].register_forward_hook(mk(tag)))
    shared_bank = np.load(REPO / "results/analysis/obs_bank.npz")
    banks = {}
    def bank_for(t):
        if a.bank == "shared":
            return shared_bank
        if t not in banks:
            banks[t] = np.load(REPO / f"results/analysis/obs_banks/t{t}.npz")
        return banks[t]
    noise = np.random.default_rng(42).standard_normal((10, 32)).astype(np.float32)
    # energy-matched donor coefficient, identical to the probe's
    def rnorm(t):
        f = fold(t); tot = 0.0
        for s_ in stems:
            if s_ not in f: continue
            b, a = f[s_]; D = b @ a
            U = USIG[s_] / torch.linalg.norm(USIG[s_], dim=0).clamp(min=1e-12)
            tot += float(((D - U @ (U.T @ D @ VS[s_]) @ VS[s_].T) ** 2).sum())
        return tot ** 0.5
    ref = 0.5 * float(np.mean([rnorm(t) for t in range(0, 76, 8)]))
    for n, (t, d) in enumerate(todo):
        t0 = time.time()
        MD = merged_delta(t, d, (ref / rnorm(d)) if d is not None else 0.0)
        with torch.no_grad():
            for s_, M in MD.items():
                if s_ not in stem2param: continue
                p = params[stem2param[s_]]
                Mx = M if tuple(p.shape) == tuple(M.shape) else M.T
                p.add_(Mx.to(p.dtype).to(p.device))
        bank = bank_for(t); n_obs = len(bank["state"])
        sig = {tag: [] for tag in HOOKS}
        for i in range(n_obs):
            policy.infer({"observation/image": bank["img"][i], "observation/wrist_image": bank["wrist"][i],
                          "observation/state": bank["state"][i], "prompt": str(bank["prompt"][i])}, noise=noise)
            missing = [tg for tg in HOOKS if tg not in acts]
            if missing: raise RuntimeError(f"no activation captured at {missing}")
            for tag in HOOKS: sig[tag].append(acts[tag].numpy().astype(np.float16))
        with torch.no_grad():
            for k, v in pristine.items(): params[k].copy_(v.to(params[k].device))
        np.savez_compressed(outdir / (f"m{t}_base.npz" if d is None else f"m{t}_{d}.npz"), **{k: np.stack(v) for k, v in sig.items()})
        print(f"[msig] {n+1}/{len(todo)}  m{t}_{d}  {time.time()-t0:.0f}s", flush=True)
    for h in handles: h.remove()
    print("[msig] done")


if __name__ == "__main__":
    main()
